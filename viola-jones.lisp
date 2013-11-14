;Viola-Jones face detector
;Training and detecting faces in pictures

(ql:quickload 'opticl)
(ql:quickload 'lparallel)
(ql:quickload 'cl-store)

(defmacro def (&rest args) `(defparameter ,@args))

(def face-dir "/media/data1000/Projects/AI_ML/facrec/faces")
(def face-desc "/media/data1000/Projects/AI_ML/facrec/faces/faces_location.txt")


(defun copy-array (array &key
                   (element-type (array-element-type array))
                   (fill-pointer (and (array-has-fill-pointer-p array)
                                      (fill-pointer array)))
                   (adjustable (adjustable-array-p array)))
  "Returns an undisplaced copy of ARRAY, with same fill-pointer and
adjustability (if any) as the original, unless overridden by the keyword
arguments."
  (let* ((dimensions (array-dimensions array))
         (new-array (make-array dimensions
                                :element-type element-type
                                :adjustable adjustable
                                :fill-pointer fill-pointer)))
    (dotimes (i (array-total-size array))
      (setf (row-major-aref new-array i)
            (row-major-aref array i)))
    new-array))

;Multidimensional array slicer
;Example of array-slice call (array-slice a '((100 200) (200 300) 2 *))
(defun array-slice (array slice)
  (let ((dim (array-dimensions array)))
    (if (and (eq (length dim) (length slice))
             (every (lambda (d r) (cond (((eq r '*) t)
                                         ((integerp r) (rangep r (list 0 d)))
                                         ((and (listp r) (eq (length r 2)) (>= (car r) (cadr r)))
                                          (and (rangep (car r) (list 0 d)) (rangep (cadr r) (list 0 d))))
                                         (t nil))))
                    dim slice))
        (let* ((N (length dim))
               (inc-index 0)
               (limit-l (make-array N))
               (limit-r (make-array N))
               (index nil)
               (result nil))
          
          (loop for d in dim
                for r in slice
                for i from 0 below (length dim) do
                (cond (((eq r '*)
                        (setf (aref limit-l i) 0)
                        (setf (aref limit-r i) d))
                       ((integerp r)
                        (setf (aref limit-l i) r)
                        (setf (aref limit-r i) (+ r 1)))
                       ((listp r)
                        (setf (aref limit-l i) (car r))
                        (setf (aref limit-r i) (+ (cadr r) 1)))
                       (t nil))))
          
          (setf index (copy-array limit-l))
          (setf result (make-array (map 'list (lambda (x y) (- x y)) limit-r limit-l)))
          
          (defun inc ()
            (incf (aref index (- N 1)))
            (when (>= (aref index (- N 1)) (aref limit-r (- N 1)))
              (setf (aref index (- N 1)) (aref limit-l (- N 1)))
              (let ((carry 1))
                (loop for i from (- N 2) downto 0
                      while (eq carry 1) do
                      (incf (aref index i))
                      (if (< (aref index i) (aref limit-r i))
                          (setf carry 0)
                        (setf (aref index i) (aref limit-l i)))))))
                      
          (loop while (< (aref index 0) (aref limit 0)) do
                (let ((dst-index (reduce #'+ (map 'vector #'-  index limit-l)))
                      (src-index (reduce #'+ index)))
                  (setf (row-major-aref result dst-index) (row-major-aref array src-index))))))))
          
                                                       
  
  

(defun load-rect (image rect)
  (let 
  t)

(defun load-image (i)
  (opticl:read-jpeg-file (join face-dir "/" "image_" (format nil "~4,'0d" i) ".jpg")))

(defun load-face-db ()
  (let* ((face-locs-file (file->string (open face-desc)))
         (face-locs (map 'vector
                         (lambda (x) (map 'vector #'floor x))
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
    (loop for i from 0 below N do
          (setf (aref result i) (load-rect (load-image i) (aref face-rects i))))
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


          
                
                
                
        