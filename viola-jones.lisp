;Viola-Jones face detector
;Training and detecting faces in pictures

;(ql:quickload 'opticl)
;(ql:quickload 'lparallel)

(defpackage :viola-jones
  (:use :cl :opticl :lparallel))

(in-package :viola-jones)

(declaim (optimize (speed 3) (safety 0)))

(setf lparallel:*kernel* (lparallel:make-kernel 5))

(defmacro def (&rest args) `(defparameter ,@args))

(def project-dir "/home/user/lisp/projects/cl-viola-jones/")
(def face-dir "/home/user/lisp/projects/cl-viola-jones/faces")
(def facedb-dir "/home/user/lisp/projects/cl-viola-jones/faces_dataset")
(def nonface-dir "/home/user/lisp/projects/cl-viola-jones/nonfaces_dataset")
(def nonface-src "/home/user/lisp/projects/cl-viola-jones/nonfaces_dataset/source.jpg")
(def face-desc "/home/user/lisp/projects/cl-viola-jones/faces/faces_location.txt")
(def *image-size* 32)

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
          (setf (aref training-labels i) 1))
    (loop for imgpath in nonfaces-paths
          for i from (length faces-paths) do
          (setf (aref training-objects i) (sum-image (opticl:read-png-file imgpath)))
          (setf (aref training-labels i) -1))
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
           (print sum)
           (print tempvars)
           (append '(lambda (img))
                    (if img-type (list (list 'declare (list 'type img-type 'img))) nil)
                    (list (list 'let (mapcar (lambda (x) (list (elt x 1) (append '(aref img) (elt x 0)))) tempvars)
                                (list '- (assoc '+ sum) (cons '+ (cdr (assoc '- sum)))))))))

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

(def *feature-cache* nil)
(def *features* nil)

;1023 of 1024, features: 71831929
;Total features: c1:3351040 c2:3351040 c3:29321600 c4:35808256
;:Target feature count: 200000
;Pruning probability: 200000/72831929 = 1/364 = 0.0027

(defmacro with-prob (p &rest args)
    `(when (< (random 1.0) ,p)
        ,@args))

(defun build-features (filepath)
    (let ((c1 0) (c2 0) (c3 0) (c4 0))
    (with-open-file (stream filepath :direction :output :if-does-not-exist :create :if-exists :overwrite)
        (loop for i from 0 below *image-size* do
            (loop for j from 0 below *image-size* do
                (format t "~s of ~s features ~s~%" (+ (* *image-size* i) j) (* *image-size* *image-size*) (+ c1 c2 c3 c4))
                (loop for h from 0 to (- *image-size* i) do
                    (loop for w from 0 to (- *image-size* j) do
                        (loop for a from 0 below w do
                            (loop for b from a below w do
                                (with-prob 0.0027
                                    (incf c3 1)
                                    (format stream "~s~%" `((+ (,i ,j ,h ,w) (,i ,(+ j b) ,h ,(- w b)))
                                                            (- (,i ,(+ j a) ,h ,(- b a))))))))
                        (loop for ti from 0 below h do
                            (loop for tj from 0 below w do
                                (with-prob 0.0027
                                    (incf c4 1)
                                    (format stream "~s~%" `((- (,i ,j ,ti ,tj) (,(+ i ti) ,(+ j tj) ,(- h ti) ,(- w tj)))
                                                            (+ (,i ,(+ j tj) ,ti ,(- w tj)) (,(+ i ti) ,j ,(- h ti) ,tj)))))))
                        (loop for a from 0 below h do
                            (with-prob 0.0027
                                (incf c1 1)
                                (format stream "~s~%" `((+ (,i ,j ,h ,w)) (- (,(+ i a) ,j ,(- h a) ,w))))))
                        (loop for a from 0 below w do
                            (with-prob 0.0027
                                (incf c2 1)
                                (format stream "~s~%" `((+ (,i ,j ,h ,w)) (- (,i ,(+ j a) ,h ,(- w a))))))))))))
    (format t "Total features c1~s c2~s c3~s c4~s~%" c1 c2 c3 c4)
    (+ c1 c2 c3 c4)))

;(defun build-feature-cache

(defun find-haar-classifier ()
    ;Error measure for weak classifiers of training process
    ;Should return numeric classification error of weak-classifier on given dataset
;    (defun error-measure (weak-classifier)
;      (loop for i from 0 below N do
;            (* (aref weights i) (abs (- (funcall weak-classifier (aref example-vecs i))
;                                        (aref example-labels i))))))

  t)


;(defun adaboost-train (example-vecs example-labels n-iter find-weak-classifier)
  ;Check arguments
;  (when (not (or (not (eq (length example-vecs) (length example-labels)))
;                 (not (equal (remove-duplicates example-labels) (vector 0 1)))
;                 (not (eq 1 (length (remove-duplicates (map 'vector #'length example-vecs)))))))
;    (print "Adaboost received incompatible arguments, aborting training")
;    (return-from adaboost-train nil))

  ;Train classifier
;  (let* ((N (length example-vecs))
;         (n-true (length (remove 0 example-labels)))
;         (n-false (length (remove 1 example-labels)))
;         (current-classifier nil)
;         (current-classifier-error 0)
;         (classifiers (make-array n-iter))
;         (betas (make-array n-iter))
;         (weights (map 'vector (lambda (x)
;                                 (if (eq x 1) (/ 1.0 (* 2.0 n-true)) (/ 1.0 (* 2.0 n-false))))
;                       example-labels)))
;  
;    (loop for i from 0 below n-iter do
;          
;          ;Normalization of weights
;          (vec-normalize weights)
;          
;          ;Find weak classifier with lowest error
;          ;Classifier is a 4-list: (fn fn-builder params error)
;          ;(funcall fn-builder params) dhould give fn
;          ;error should be (error-measure fn)
;          (setf current-classifier (funcall #'find-weak-classfier example-vecs example-labels))
;          (setf (aref classifiers i) current-classifier)
;          (setf current-classifier-error (elt current-classifier 3))
;          
;          ;Save beta
;          (setf (aref betas i)
;                (/ current-classifier-error (- 1.0 current-classifier-error)))
;          
;          ;Update weights
;          (loop for j from 0 below N do
;                (let ((beta (aref betas i))
;                      (example-error (abs (- (aref example-labels i)
;                                             (funcall (car current-classifier) (aref example-vecs i))))))
;                  (setf (aref weights j) (* (aref weights j)
;                                            (expt beta (- 1.0 example-error)))))))
;    
;    ;Return strong classifier as vector
;    (vector (lambda (x)
;              (let ((left-sum 0)
;                    (right-sum 0))
;                (loop for i from 0 below n-iter do
;                      (let ((alpha (log (/ 1.0 (aref betas i)))))
;                        (incf left-sum (* alpha  (funcall (car (aref classifiers i)) x)))
;                        (incf right-sum alpha))
;                      (if (>= left-sum (* 0.5 right-sum))
;                        1
;                        0))))
;            classifiers
;            betas)))


          
                
                
                
        
