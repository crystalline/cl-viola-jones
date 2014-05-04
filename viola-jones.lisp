;Viola-Jones face detector
;Training and detecting faces in pictures

;(ql:quickload 'opticl)
;(ql:quickload 'lparallel)

(defpackage :viola-jones
  (:use :cl :opticl :lparallel))

(in-package :viola-jones)

(setf lparallel:*kernel* (lparallel:make-kernel 5))

(defmacro def (&rest args) `(defparameter ,@args))

(def face-dir "/media/data1000/Projects/AI_ML/facerec/faces")
(def face-desc "/media/data1000/Projects/AI_ML/facerec/faces/faces_location.txt")

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
                                 :initial-element (floor (img-mean image)))))
        (image-blit (opticl:resize-image image (elt newdim 0) (elt newdim 1))
                    result
                    (floor (/ (- side (elt newdim 0)) 2))
                    (floor (/ (- side (elt newdim 1)) 2)))))))

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

(defun load-rect (image rect)
  (let* ((slice (transform-rect rect 0.1 0.1 35 0))
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
          (scale-to-square result 64))
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

(defun load-face-db (&optional (range '(0 450)))
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

(defun prepare-face-db (folder &optional (range '(0 450)))
  (def faces (load-face-db range))
  (loop for img across faces
        for i from 0 do
        (when (arrayp img)
          (write-png-file (pathname
                           (join "/media/data1000/Projects/AI_ML/facerec/"
                                 folder (format nil "//~s.png" i))) img))))

(defun sum-image (img)
  (let* ((dim (array-dimensions img))
         (height (elt dim 0))
         (width (elt dim 1))
         (sum 0)
         (result (make-array dim)))
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

;Rect is #(y x h w)
(defun haar-horiz (image y x h w a theta)
  (if (> (- (+ (aref image y x)
               (aref image (+ y h) (+ x w)))
            (+ (aref image y (+ x w))
               (aref image (+ y h) w)))
         theta)
      1 -1))
     
  

(defun find-haar-classifier (error-fn)
    ;Error measure for weak classifiers of training process
    ;Should return numeric classification error of weak-classifier on given dataset
    (defun error-measure (weak-classifier)
      (loop for i from 0 below N do
            (* (aref weights i) (abs (- (funcall weak-classifier (aref example-vecs i))
                                        (aref example-labels i))))))

  t)


(defun adaboost-train (example-vecs example-labels n-iter find-weak-classifier)
  ;Check arguments
  (when (not (or (not (eq (length example-vecs) (length example-labels)))
                 (not (equal (remove-duplicates example-labels) (vector 0 1)))
                 (not (eq 1 (length (remove-duplicates (map 'vector #'length example-vecs)))))))
    (print "Adaboost received incompatible arguments, aborting training")
    (return-from adaboost-train nil))
  
  ;Train classifier
  (let* ((N (length example-vecs))
         (n-true (length (remove 0 example-labels)))
         (n-false (length (remove 1 example-labels)))
         (current-classifier nil)
         (current-classifier-error 0)
         (classifiers (make-array n-iter))
         (betas (make-array n-iter))
         (weights (map 'vector (lambda (x)
                                 (if (eq x 1) (/ 1.0 (* 2.0 n-true)) (/ 1.0 (* 2.0 n-false))))
                       example-labels)))
  
    (loop for i from 0 below n-iter do
          
          ;Normalization of weights
          (vec-normalize weights)
          
          ;Find weak classifier with lowest error
          ;Classifier is a 4-list: (fn fn-builder params error)
          ;(funcall fn-builder params) dhould give fn
          ;error should be (error-measure fn)
          (setf current-classifier (funcall #'find-weak-classfier example-vecs example-labels))
          (setf (aref classifiers i) current-classifier)
          (setf current-classifier-error (elt current-classifier 3))
          
          ;Save beta
          (setf (aref betas i)
                (/ current-classifier-error (- 1.0 current-classifier-error)))
          
          ;Update weights
          (loop for j from 0 below N do
                (let ((beta (aref betas i))
                      (example-error (abs (- (aref example-labels i)
                                             (funcall (car current-classifier) (aref example-vecs i))))))
                  (setf (aref weights j) (* (aref weights j)
                                            (expt beta (- 1.0 example-error)))))))
    
    ;Return strong classifier as vector
    (vector (lambda (x)
              (let ((left-sum 0)
                    (right-sum 0))
                (loop for i from 0 below n-iter do
                      (let ((alpha (log (/ 1.0 (aref betas i)))))
                        (incf left-sum (* alpha  (funcall (car (aref classifiers i)) x)))
                        (incf right-sum alpha))
                      (if (>= left-sum (* 0.5 right-sum))
                        1
                        0))))
            classifiers
            betas)))


          
                
                
                
        
