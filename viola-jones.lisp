;Viola-Jones face detector
;Training and detecting faces in pictures

;(ql:quickload 'opticl)
;(ql:quickload 'lparallel)

(defpackage :viola-jones
  (:use :cl :opticl :lparallel))

(in-package :viola-jones)

;(declaim (optimize (speed 3) (safety 0)))

(setf lparallel:*kernel* (lparallel:make-kernel 5))

(def project-dir "/home/user/lisp/projects/cl-viola-jones/")
(def face-dir "/home/user/lisp/projects/cl-viola-jones/faces")
(def facedb-dir "/home/user/lisp/projects/cl-viola-jones/faces_dataset")
(def nonface-dir "/home/user/lisp/projects/cl-viola-jones/nonfaces_dataset")
(def nonface-src "/home/user/lisp/projects/cl-viola-jones/nonfaces_dataset/source.jpg")
(def face-desc "/home/user/lisp/projects/cl-viola-jones/faces/faces_location.txt")
(def *image-size* 32)

(def label-face 1)
(def label-nonface 0)

(def training-objects nil)
(def training-labels nil)

(defun reduce-image (fn start img)
  (let* ((dim (array-dimensions img))
         (height (elt dim 0))
         (width (elt dim 1))
         (sum start))
  (loop for i from 0 below height do
        (loop for j from 0 below width do
              (setf sum (funcall fn sum (aref img i j)))))
  sum))

(defun img-mean (img)
  (let ((N (reduce #'* (array-dimensions img))))
    (/ (reduce-image #'+ 0 img) N)))

(defun image-blit (src dst off-y off-x)
  (let* ((dim (array-dimensions src))
         (height (elt dim 0))
         (width (elt dim 1)))
    (loop for i from 0 below height do
          (loop for j from 0 below width do
                (setf (aref dst (+ i off-y) (+ j off-x))
                      (aref src i j))))
    dst))

(defun scale-to-square (image side)
  (let ((dim (array-dimensions image)))
    (if (> (length dim) 2)
        (progn
          (print "Error! Not an image")
          nil)
      (let* ((maxdim (apply #'max dim))
             (newdim (mapcar (lambda (x) (* x (/ side maxdim))) dim))
             (result (make-array (list side side)
                                 :element-type (array-element-type image)
                                 :initial-element (floor (img-mean image))))
             (transformed
              (image-blit (opticl:resize-image image (elt newdim 0) (elt newdim 1))
                          result
                          (floor (/ (- side (elt newdim 0)) 2))
                          (floor (/ (- side (elt newdim 1)) 2)))))
                 transformed))))

(defun rect->square (rect)
  (let* ((i-low (elt rect 3))
         (i-high (elt rect 1))
         (j-low (elt rect 0))
         (j-high (elt rect 2))
         (h (- i-high i-low))
         (w (- j-high j-low))
         (cx (floor (/ (+ j-low j-high) 2)))
         (cy (floor (/ (+ i-low i-high) 2))))
    (if (> h w)
        (list (- cx (floor (/ h 2))) i-high (+ cx (floor (/ h 2))) i-low)
      (list j-low (+ cy (floor (/ w 2))) j-high (- cy (floor (/ w 2)))))))

(defun transform-rect (rect alpha beta delta-i delta-j)
  (let* ((i-low (elt rect 3))
         (i-high (elt rect 1))
         (j-low (elt rect 0))
         (j-high (elt rect 2))
         (h (- i-high i-low))
         (w (- j-high j-low))
         (dh (floor (* alpha h)))
         (dw (floor (* beta w))))
    (list (list (+ i-low dh delta-i) (- (+ i-high delta-i) dh))
          (list (+ j-low dw delta-j) (- (+ j-high delta-j) dw)))))

(defun image-window (image y x h w)
    (let ((res (make-array (list h w)
                                 :element-type (array-element-type image))))
        (loop for is from y to (- (+ y h) 1)
              for i from 0 do
            (loop for js from x to (- (+ x w) 1)
                  for j from 0 do
                (setf (aref res i j) (aref image is js))))
        res))

(defun load-rect (image rect)
  (let* ((slice (transform-rect (rect->square rect) 0.19 0.19 35 0))
         (img (array-slice image (append slice (list '*)))))
    (if img
        (let* ((dim (array-dimensions img))               (result (make-array (list (elt dim 0) (elt dim 1))
                                   :element-type (array-element-type img))))
          (loop for i from 0 below (elt dim 0) do
                (loop for j from 0 below (elt dim 1) do
                      (let ((sum 0))
                        (loop for k from 0 below (elt dim 2) do
                              (incf sum (aref img i j k)))
                        (setf (aref result i j) (floor (/ sum 3))))))
          (scale-to-square result *image-size*))
      (progn (print "Error, image rect is ot of range")
             nil))))

(defun load-image (i)
  (let ((iname (join face-dir "/" "image_" (format nil "~4,'0d" i) ".jpg")))
    (format t "Loading image ~s" iname)
    (let ((res (opticl:read-jpeg-file iname)))
      (if res
          (progn ;(format t "Dim: ~a" (array-dimensions res))
                 res)
        (progn (format t "Error! Empty file ~s" iname)
               nil)))))

(defun extract-faces (&optional (range '(0 450)))
  (let* ((face-locs-file (file->string (open face-desc)))
         (face-locs (map 'vector
                         (lambda (x) (map 'vector (lambda (elem) (abs (floor elem))) x))
                         (read-from-string
                          (join
                           "("
                           (map 'string (lambda (c)
                                          (let ((rep (assoc c '((#\; #\ )
                                                                (#\[ #\( )
                                                                (#\] #\) )))))
                                            (if rep (cadr rep) c)))
                                face-locs-file)
                           ")"))))
         (face-rects (map 'vector (lambda (x) (vector (aref x 0) (aref x 1)
                                                      (aref x 4) (aref x 5)))
                          face-locs))
         (N (length face-rects))
         (result (make-array N)))
    (print "Loading image database")
    (loop for i from (elt range 0) below (elt range 1) do
          (print  (aref face-rects i))
          (setf (aref result i) (load-rect (load-image (+ i 1)) (aref face-rects i))))
    result))

(defun load-training-dataset (faces-folder nonfaces-folder)
  (let* ((faces-paths (directory (join faces-folder "/*.png")))
         (nonfaces-paths (directory (join nonfaces-folder "/*.png")))
         (n (+ (length faces-paths) (length nonfaces-paths))))
    (format t "Loading dataset. Faces: ~s (~s files) Nonfaces: ~s (~s files)"
              faces-folder (length faces-paths) nonfaces-folder  (length nonfaces-paths))
    (setf training-objects (make-array n))
    (setf training-labels (make-array n))
    (loop for imgpath in faces-paths
          for i from 0 do
          (setf (aref training-objects i) (sum-image (opticl:read-png-file imgpath)))
          (setf (aref training-labels i) label-face))
    (loop for imgpath in nonfaces-paths
          for i from (length faces-paths) do
          (setf (aref training-objects i) (sum-image (opticl:read-png-file imgpath)))
          (setf (aref training-labels i) label-nonface))
    t))

;Make n random ph x pw patches from input image and return them as a vector
(defun make-image-patches (input-image n p-height p-width)
  (let ((res (make-array n))
          (height (car (array-dimensions input-image)))
        (width (cadr (array-dimensions input-image))))
     (loop for i from 0 to (- n 1) do
         (setf (aref res i)
               (image-window input-image (random (- height p-height)) (random (- width p-width))
                                           p-height p-width)))
     res))

(defun build-nonface-db (src folder)
  (let ((srcimg (opticl::convert-image-to-grayscale (opticl:read-jpeg-file src))))
      (loop for img across (make-image-patches srcimg 450 *image-size* *image-size*)
            for i from 0 do
            (write-png-file (pathname
                             (join "/home/user/lisp/projects/cl-viola-jones/"
                                   folder (format nil "//~s.png" i))) img))))

;(build-nonface-db nonface-src "nonfaces")

;Build array of side x side grayscale face images and save them to folder as 0.png ... [n].png
(defun build-face-db (folder &optional (range '(0 450)))
  (let ((faces (extract-faces range)))
    (loop for img across faces
          for i from 0 do
          (when (arrayp img)
            (write-png-file (pathname
                             (join "/home/user/lisp/projects/cl-viola-jones/"
                                   folder (format nil "//~s.png" i))) img)))))

(defun sum-image (img)
  (let* ((dim (array-dimensions img))
         (height (elt dim 0))
         (width (elt dim 1))
         (sum 0)
         (result (make-array dim :element-type 'fixnum)))
    (loop for i from 0 below height do
          (loop for j from 0 below width do
                (incf sum (aref img i j))
                (setf (aref result i j) sum)))
    result))
    
(defun vec-normalize (vec)
  (let ((sum (reduce #'+ vec)))
    (loop for i from 0 to (- (length vec) 1) do
          (setf (aref vec i) (/ (aref vec i) sum)))
    vec))

;Weak classifiers

;Rect is #(y x h w)
(declaim (inline isum))
(defun isum (image y x h w)
  (- (aref image (+ y h) (+ x w))
     (- (+ (aref image y (+ x w))
           (aref image (+ y h) x))
        (aref image y x))))

;(defmacro def-haar-feature (param-lst rect-lst)
;  (

;(def-haar-feature ((params y x w h a)
;                   (rects (y x (+ x a) h) (y x (- w a) h))))

;(defun make-haar-feature (ftype &rest args)

;(make-haar-feature ((type (SIMPLE-ARRAY FIXNUM (32 32))) (w 32) (h 32) (+ (10 10 4 4)) (- (10 14 4 4)))
; ...................^width ^height ^rect: y x h w (matrix-like notation)

(defmacro filter (&rest args) `(remove-if-not ,@args))

(defun gen-refs (rect)
  (destructuring-bind (y x h w) rect
    (list (list '- (list (+ y h) x)
                   (list y (+ x w)))
          (list '+ (list (+ y h) (+ x w))
                     (list y x)))))

;(apply #'append (mapcar #'cdr (filter (lambda (x) (eq '- (car x)))
(defun make-haar-feature (spec)
    (let* (;(w (if (assoc 'w spec) *image-size*))
           ;(h (if (assoc 'h spec) *image-size*))
           (positive-label (if (assoc 'positive-label spec) (cadr (assoc 'positive-label spec)) label-face))
           (negative-label (if (assoc 'negative-label spec) (cadr (assoc 'negative-label spec)) label-nonface))
           (theta (cadr (assoc 'theta spec)))
           (parity (cadr (assoc 'parity spec)))
           (img-type (cadr (assoc 'type spec)))
           (refs (append (apply #'append (mapcar #'gen-refs (cdr (assoc '+ spec))))
                         (mapcar (lambda (x) (if (eq '+ (car x)) (cons '- (cdr x)) (cons '+ (cdr x))))
                                 (apply #'append (mapcar #'gen-refs (cdr (assoc '- spec)))))))
           (sum (list (cons '+ (apply #'append (mapcar (lambda (x) (if (eq '+ (car x)) (cdr x) nil)) refs)))
                         (cons '- (apply #'append (mapcar (lambda (x) (if (eq '- (car x)) (cdr x) nil)) refs)))))
           (symcount 0)
           (tempvars nil))
           (loop for ref in (append (cdr (assoc '+ sum)) (cdr (assoc '- sum))) do
             (when (not (assoc ref tempvars :test #'equal))
                 (push (list ref (intern (string-upcase (format nil "ref~s" symcount)))) tempvars)
                 (incf symcount)))
           (setf sum (tree-replace tempvars sum))
           ;(print sum)
           ;(print tempvars)
           (append '(lambda (img))
                    (if img-type (list (list 'declare (list 'type img-type 'img))) nil)
                    (list (list 'let (mapcar (lambda (x) (list (elt x 1) (append '(aref img) (elt x 0)))) tempvars)
                                (let ((feature-expr (list '- (assoc '+ sum) (cons '+ (cdr (assoc '- sum))))))
                                  (if (every #'numberp (list parity theta))
                                      (if (> parity 0)
                                        (list 'if (list '> feature-expr theta) positive-label negative-label)
                                        (list 'if (list '> feature-expr theta) negative-label positive-label))
                                      feature-expr)))))))

;Test:
;(eval (make-haar-feature (cons (list 'type (type-of (elt training-objects 0))) '((w 32) (h 32) (+ (10 10 4 4)) (- (10 14 4 4))))))

(defun haar-horiz (image y x h w a)
  (- (isum image y x h a)
     (isum image y (+ x a) h (- w a))))

(defun haar-vert (image y x h w a)
  (- (isum image y x a w)
     (isum image (+ y a) x (- h a) w)))

(defun haar-wide (image y x h w a b)
  (- (+ (isum image y x h a)
        (isum image y (+ x b) h (- w (+ a b))))
     (isum image y (+ x a) h (- b a))))

(defun haar-square (image y x h w dy dx)
  (- (+ (isum image y (+ x dx) dy (- w dx))
        (isum image (+ y dy) x (- h dy) dx))
     (+ (isum image y x dy dx)
        (isum image (+ y dy) (+ x dx) (- h dy) (- w dx)))))

(defun count-diff (vec1 vec2)
  (if (not (eq (length vec1) (length vec2)))
      nil
    (let ((diff 0))
      (loop for i across vec1
            for j across vec2 do
            (when (not (eq i j))
              (incf diff 1)))
      diff)))

(defun find-best-threshold (objects labels fn &rest args)
  (let* ((features (make-array (length objects)))
         (fsum+1 0)
         (fsum-1 0)
         (fmean+1 0)
         (fmean-1 0)
         (N (length objects))
         (invn (/ 1 (float N)))
         (theta 0)
         (parity 1))
    (loop for object across objects
          for label across labels
          for i from 0 to (- N 1) do
          ;Calculate feature value
          (let ((f (apply fn (cons object args))))
            (if (eq label 1)
                (incf fsum+1 f)
              (incf fsum-1 f))
            (setf (elt features i) f)))
    ;Mean feature values for classes
    (setf fmean+1 (* fsum+1 invn))
    (setf fmean-1 (* fsum-1 invn))
    (format t "fmean+1 ~s fmean-1 ~s" fmean+1 fmean-1)
    ;Choose threshold and parity
    (if (> fmean+1 fmean-1)
        (setf parity 1)
      (setf parity -1))
    ;TODO: threshold modulation to acheive favorable false positive pattern
    ;(if (> 0.001 (abs (- fmean+1 fmean-1)))
    (setf theta (/ (+ fmean+1 fmean-1) 2))
    (let ((feature-error
           (* invn (count-diff
            ;Weak classifier's guess
            (if (eq parity 1)
                (map 'vector (lambda (f) (if (> f theta) 1 -1)) features)
              (map 'vector (lambda (f) (if (< f theta) 1 -1)) features))
            ;Ground truth
            labels))))
      (list (list 'parity parity)
            (list 'theta theta)
            (list 'error feature-error)))))

;Total features: 71831929
;features: c1:3351040 c2:3351040 c3:29321600 c4:35808256
;Target feature count: 200000
;Pruning probability: 200000/72831929 = 1/364 = 0.0027

(defmacro with-prob (p &rest args)
    `(when (< (random 1.0) ,p)
        ,@args))

(def feature-specs nil)
(def feature-lambdas nil)
(def feature-image-cache nil)

(def *prune-p* 0.0001)

(defun build-features ();(filepath)
    (let ((c1 0) (c2 0) (c3 0) (c4 0)
          (flist nil))
    ;(with-open-file (stream filepath :direction :output :if-does-not-exist :create :if-exists :overwrite)
        (loop for i from 0 below *image-size* do
            (loop for j from 0 below *image-size* do
                (format t "~s of ~s features ~s~%" (+ (* *image-size* i) j) (* *image-size* *image-size*) (+ c1 c2 c3 c4))
                (loop for h from 0 to (- *image-size* i 1) do
                    (loop for w from 0 to (- *image-size* j 1) do
                        (loop for a from 0 below w do
                            (loop for b from a below w do
                                (with-prob *prune-p*
                                    (incf c3 1)
                                    (push `((+ (,i ,j ,h ,w) (,i ,(+ j b) ,h ,(- w b)))
                                            (- (,i ,(+ j a) ,h ,(- b a)))) flist))))
                                    ;(format stream "~s~%" `((+ (,i ,j ,h ,w) (,i ,(+ j b) ,h ,(- w b)))
                                    ;                        (- (,i ,(+ j a) ,h ,(- b a))))))))
                        (loop for ti from 0 below h do
                            (loop for tj from 0 below w do
                                (with-prob *prune-p*
                                    (incf c4 1)
                                    (push `((- (,i ,j ,ti ,tj) (,(+ i ti) ,(+ j tj) ,(- h ti) ,(- w tj)))
                                            (+ (,i ,(+ j tj) ,ti ,(- w tj)) (,(+ i ti) ,j ,(- h ti) ,tj))) flist))))
                                    ;(format stream "~s~%" `((- (,i ,j ,ti ,tj) (,(+ i ti) ,(+ j tj) ,(- h ti) ,(- w tj)))
                                    ;                        (+ (,i ,(+ j tj) ,ti ,(- w tj)) (,(+ i ti) ,j ,(- h ti) ,tj)))))))
                        (loop for a from 0 below h do
                            (with-prob *prune-p*
                                (incf c1 1)
                                ;(format stream "~s~%" `((+ (,i ,j ,h ,w)) (- (,(+ i a) ,j ,(- h a) ,w))))))
                                (push `((+ (,i ,j ,h ,w)) (- (,(+ i a) ,j ,(- h a) ,w))) flist)))
                        (loop for a from 0 below w do
                            (with-prob *prune-p*
                                (incf c2 1)
                                ;(format stream "~s~%" `((+ (,i ,j ,h ,w)) (- (,i ,(+ j a) ,h ,(- w a))))))))))))
                                (push `((+ (,i ,j ,h ,w)) (- (,i ,(+ j a) ,h ,(- w a)))) flist)))))))
    ;Convert to vector for fast access
    (setf feature-specs (make-array (length flist)))
    (loop for item in flist
          for i from 0 do (setf (aref feature-specs i) item))
    
    ;Compile features
    (setf feature-lambdas (make-array (length feature-specs)))
    
    (let ((count 0))
        (loop for i from 0 below (length feature-specs) do
           (setf (aref feature-lambdas i) (eval (make-haar-feature (aref feature-specs i))))
           (incf count)
           (print count)))
    (format t "Total features c1 ~s c2 ~s c3 ~s c4 ~s~%" c1 c2 c3 c4)
    (+ c1 c2 c3 c4)))

(load-training-dataset facedb-dir nonface-dir)

(defun update-sum-cache (weights)
    (loop for elem across feature-image-cache do
        (let ((fvals (elt elem 0))
              (S+ (elt elem 1))
              (S- (elt elem 2)))
           ;Zero element
           (if (> (aref training-labels (cdr (aref fvals 0))) 0.5)
                  (progn (setf (aref S+ 0) (aref weights 0))
                         (setf (aref S- 0) 0))
                  (progn (setf (aref S+ 0) 0)
                         (setf (aref S- 0) (aref weights 0))))
           ;Rest
           (loop for i from 1 below (length weights) do
               (if (> (aref training-labels (cdr (aref fvals i))) 0.5)
                  (progn (setf (aref S+ i) (+ (aref weights (cdr (aref fvals i))) (aref S+ (- i 1))))
                         (setf (aref S- i) (aref S+ (- i 1))))
                  (progn (setf (aref S+ i) (aref S+ (- i 1)))
                         (setf (aref S- i) (+ (aref weights (cdr (aref fvals i))) (aref S- (- i 1))))))))))

(def start-weights (map 'vector (lambda (x) x) (build-list (length training-objects) (lambda (i) (/ 1.0 (length training-objects))))))

(defun build-feature-cache ()
    (let ((N (length training-objects))
          (weights (map 'vector (lambda (x) x) (build-list (length training-objects) (lambda (i) (/ 1.0 (length training-objects)))))))
        (setf feature-image-cache (make-array (length feature-lambdas)))
        (loop for fn across feature-lambdas
              for i from 0 do
            (format t "Building feature cache for feature #~s~%" i)
            (setf (aref feature-image-cache i) (vector (sort (map 'vector #'cons (map 'vector (aref feature-lambdas i) training-objects)    ;Pairs of Feature value ( Object ) : Index
                                                                                 (build-list N (lambda (i) i)))
                                                       (lambda (x y) (<= (car x) (car y))))
                                               (make-array N :element-type 'float :initial-element 0.0)   ;Sum of + labeled example's weights below current element
                                               (make-array N :element-type 'float :initial-element 0.0)))) ;Sum of - labeled example's weights below current element
    (update-sum-cache weights)))

(defun validate-classifier (fn)
    (let ((err (/ (float (count-diff (map 'vector (elt res 0) training-objects) training-labels))
                  (length training-objects))))
       (format t "Validating classifier, ~s examples, total error: ~s~%"
               (length training-objects) err)))

(defun find-best-haar-classifier (weights)
    (update-sum-cache weights)
    (let ((min-error 1.0)
          (min-feature nil)
          (theta 0)
          (parity 1)
          (n (length training-objects)))
        (loop for f across feature-image-cache
              for i from 0 do
           (let ((fvals (elt f 0))
                 (S+ (elt f 1))
                 (S- (elt f 2))
                 (T+ (aref (elt f 1) (- n 1)))
                 (T- (aref (elt f 2) (- n 1))))
             (loop for j below n do
                (let ((err (+ (aref S+ j) (- T- (aref S- j)))))
                    (when (> min-error err)
                      ;(print err)
                      (setf min-error err)
                      (setf parity 1)
                      (setf theta (car (aref fvals j)))
                      (setf min-feature (append (list '(parity 1) (list 'theta theta)) (aref feature-specs i)))))
                (let ((err (+ (aref S- j) (- T+ (aref S+ j)))))
                    (when (> min-error err)
                      ;(print err)
                      (setf min-error err)
                      (setf parity -1)
                      (setf theta (car (aref fvals j)))
                      (setf min-feature (append (list '(parity -1) (list 'theta theta)) (aref feature-specs i))))))))
        (print min-error)
        (list min-error min-feature (eval (make-haar-feature min-feature)))))

(defun haar-adaboost-train (example-vecs example-labels n-iter find-weak-classifier)
  ;Check arguments
  (when (not (or (not (eq (length example-vecs) (length example-labels)))
                 (not (equal (remove-duplicates example-labels) (vector 0 1)))
                 (not (eq 1 (length (remove-duplicates (map 'vector #'length example-vecs)))))))
    (print "Adaboost received incompatible arguments, aborting training")
    (return-from haar-adaboost-train nil))

  ;Train classifier
  (let* ((N (length example-vecs))
         (n-true (length (remove 0 example-labels)))
         (n-false (length (remove 1 example-labels)))
         (current-classifier nil)
         (current-classifier-error 0)
         (current-classifier-fn nil)
         (classifiers (make-array n-iter))
         (betas (make-array n-iter))
         (weights (map 'vector (lambda (x)
                                 (if (eq x 1) (/ 1.0 (* 2.0 n-true)) (/ 1.0 (* 2.0 n-false))))
                       example-labels)))
  
    (loop for i from 0 below n-iter do
          
          ;Normalization of weights
          (vec-normalize weights)
          
          ;Find weak classifier with lowest error
          (setf current-classifier (funcall find-weak-classifier weights))
          (setf current-classifier-fn (elt current-classifier 2))
          (setf (aref classifiers i) current-classifier)
          (setf current-classifier-error (elt current-classifier 0))
          
          ;Save beta
          (setf (aref betas i)
                (/ current-classifier-error (- 1.0 current-classifier-error)))
          
          ;Update weights
          (loop for j from 0 below N do
                (let ((beta (aref betas i))
                      (example-error (abs (- (aref example-labels j)
                                             (funcall current-classifier-fn (aref example-vecs j))))))
                  (setf (aref weights j) (* (aref weights j)
                                            (expt beta (- 1.0 example-error)))))))
    
    ;Return strong classifier as vector
    (vector (lambda (x)
              (let ((left-sum 0)
                    (right-sum 0))
                (loop for i from 0 below n-iter do
                      (let ((alpha (log (/ 1.0 (aref betas i)))))
                        (incf left-sum (* alpha (funcall (elt (aref classifiers i) 2) x)))
                        (incf right-sum alpha)))
                (if (>= left-sum (* 0.5 right-sum))
                    1
                    0)))
            classifiers
            betas)))


          
                
                
                
        
