;Viola-Jones face detector
;Training and detecting faces in pictures

(ql:quickload 'opticl)
(ql:quickload 'lparallel)

(setf lparallel:*kernel* (lparallel:make-kernel 5))

(load "/media/data1000/Projects/AI_ML/facerec/util.lisp")

(defmacro def (&rest args) `(defparameter ,@args))

(def face-dir "/media/data1000/Projects/AI_ML/facerec/faces")
(def face-desc "/media/data1000/Projects/AI_ML/facerec/faces/faces_location.txt")

(defun load-rect (image rect)
  (let ((img (array-slice image (list (list (elt rect 3) (elt rect 1))
                                      (list (elt rect 0) (elt rect 2))
                                      '*))))
    (if img
        (let* ((dim (array-dimensions img))
               (result (make-array (list (elt dim 0) (elt dim 1)))))
          (loop for i from 0 below (elt dim 0) do
                (loop for j from 0 below (elt dim 1) do
                      (let ((sum 0))
                        (loop for k from 0 below (elt dim 2) do
                              (incf sum (aref img i j k)))
                        (setf (aref result i j) (floor (/ sum 3))))))
          result)
      (progn (print "Error, image rect is ot of range")
             nil))))

(defun load-image (i)
  (let ((iname (join face-dir "/" "image_" (format nil "~4,'0d" i) ".jpg")))
    (format t "Loading image ~s" iname)
    (let ((res (opticl:read-jpeg-file iname)))
      (if res
          (progn (format t "Dim: ~a" (array-dimensions res))
                 res)
        (progn (format t "Error! Empty file ~s" iname)
               nil)))))

(defun load-face-db ()
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
;    (lparallel:pmap 'vector (lambda (i)  (load-rect (load-image (+ i 1)) (aref face-rects i)))
;          (build-list N (lambda (x) x)))))
    (loop for i from 249 below N do
          (print  (aref face-rects i))
          (setf (aref result i) (load-rect (load-image (+ i 1)) (aref face-rects i))))
    result))

(defun vec-normalize (vec)
  (let ((sum (reduce #'+ vec)))
    (loop for i from 0 to (- (length vec) 1) do
          (setf (aref vec i) (/ (aref vec i) sum)))
    vec))

(defun find-haar-classifier (error-fn)
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
    
    ;Error measure for weak classifiers of training process
    ;Should return numeric classification error of weak-classifier on given dataset
    (defun error-measure (weak-classifier)
      (loop for i from 0 below N do
            (* (aref weights i) (abs (- (funcall weak-classifier (aref example-vecs i))
                                        (aref example-labels i))))))
            
    (loop for i from 0 below n-iter do
          
          ;Normalization of weights
          (vec-normalize weights)
          
          ;Find weak classifier with lowest error
          ;Classifier is a 4-list: (fn fn-builder params error)
          ;(funcall fn-builder params) dhould give fn
          ;error should be (error-measure fn)
          (setf current-classifier (funcall #'find-weak-classfier #'error-measure))
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


          
                
                
                
        