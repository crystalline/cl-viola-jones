(defsystem viola-jones
  :name "viola-jones"
  :description "Object detection with Viola-Jones framework"
  :license "MIT"
  :author "Crystalline Emerald"
  :version "0.1"
  :depends-on (:opticl :lparallel)
  :serial nil
  :components ((:file "viola-jones") (:file "util")))
