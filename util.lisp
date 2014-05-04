
(in-package :viola-jones)

;Utilities for IO and array operations
(defun file->string (path)
  (with-open-file (stream path)
    (let ((data (make-string (file-length stream))))
      (read-sequence data stream)
      data)))

(defun join (&rest args)
  (apply #'concatenate 'string args))

(defun rangep (x range)
  (and (>= x (car range)) (<= x (cadr range))))

(defun build-list (N fn)
  (let ((res nil))
    (loop for i from 0 below N do
          (push (funcall fn i) res))
    res))

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
             (every (lambda (d r) (cond ((eq r '*) t)
                                        ((integerp r) (and (>= r 0) (< r d)))
                                        ((and (listp r) (eq (length r) 2) (<= (car r) (cadr r)))
                                         (and (>= (car r) 0) (< (car r) d) (>= (cadr r) 0) (< (cadr r) d)))
                                        (t (print "Invalid description, aborting slice") nil)))
                    dim slice))
        (let* ((N (length dim))
               (limit-l (make-array N))
               (limit-r (make-array N))
               (loop-flag t)
               (index nil)
               (result nil))
          
          (loop for d in dim
                for r in slice
                for i from 0 below (length dim) do
                (cond ((eq r '*)
                        (setf (aref limit-l i) 0)
                        (setf (aref limit-r i) d))
                       ((integerp r)
                        (setf (aref limit-l i) r)
                        (setf (aref limit-r i) (+ r 1)))
                       ((listp r)
                        (setf (aref limit-l i) (car r))
                        (setf (aref limit-r i) (+ (cadr r) 1)))
                       (t nil)))
          
          (setf index (copy-array limit-l))
          (setf result (make-array (map 'list (lambda (x y) (- x y)) limit-r limit-l)
                                   :element-type (array-element-type array)))
          (print (array-element-type array))
          
          (defun inc ()
            (incf (aref index (- N 1)))
            (when (>= (aref index (- N 1)) (aref limit-r (- N 1)))
              (if (eq N 0)
                  (setf loop-flag nil)
                  (setf (aref index (- N 1)) (aref limit-l (- N 1))))
              (let ((carry 1))
                (loop for i from (- N 2) downto 0
                      while (eq carry 1) do
                      (incf (aref index i))
                      (if (< (aref index i) (aref limit-r i))
                          (setf carry 0)
                        (if (eq i 0)
                            (setf loop-flag nil)
                            (setf (aref index i) (aref limit-l i))))))))
                      
          (loop while loop-flag do
                (let ((dst-index (apply #'array-row-major-index result (map 'list #'- index limit-l)))
                      (src-index (apply #'array-row-major-index
                                        array
                                        (map 'list (lambda(x)x) index))))
                  (setf (row-major-aref result dst-index) (row-major-aref array src-index)))
                (inc))
          
          result))))

(defun test-slicer ()
  (let ((N 4))
    (def a (make-array (list N N N)))
    (loop for i from 0 below N do
          (loop for j from 0 below N do
                (loop for k from 0 below N do
                      (setf (aref a i j k) (+ i j k)))))
    (def b (array-slice a (list '* '* '*)))
    (print (array-dimensions a))
    (print (array-dimensions b))
    (print a)
    (print b)
    (format t "Equality test: ~s~%" (equal a b))))
