;add negative reward to respond-unconfident
(clear-all)

(define-model v6-no-adapt

(sgp :seed (123456 0))
(sgp :v nil :show-focus t)
(sgp :esc t :bll 0.5 :ol t :er t :mas 5) ; :lf can be adjusted in python script
(sgp :egs 3 :ul t :ult nil)
(sgp :record-ticks t :time-noise 0.015)
  
(chunk-type read-letters step response block alpha confident count beta)
(chunk-type stimulus type duration)
(chunk-type threshold value)

(add-dm 
 (start isa chunk) (attend isa chunk) (compare isa chunk) (recall isa chunk)
 (respond isa chunk) (done isa chunk) (encode isa chunk) (wait isa chunk)
 (standard isa chunk) (oddball isa chunk) (uncertain isa chunk)
 (increment isa chunk) (decrement isa chunk) (yes isa chunk) (no isa chunk)

 (gd0 isa read-letters step start alpha 0 block decrement count 0 beta nil)
 (gd1 isa read-letters step start alpha 1 block decrement count 0 beta nil)
 (gd2 isa read-letters step start alpha 2 block decrement count 0 beta nil)
 (gd3 isa read-letters step start alpha 3 block decrement count 0 beta nil)
 (gd4 isa read-letters step start alpha 4 block decrement count 0 beta nil)
 (gd5 isa read-letters step start alpha 5 block decrement count 0 beta nil)
 (gd6 isa read-letters step start alpha 6 block decrement count 0 beta nil)
 (gd7 isa read-letters step start alpha 7 block decrement count 0 beta nil)
 (gd8 isa read-letters step start alpha 8 block decrement count 0 beta nil)
 (gd9 isa read-letters step start alpha 9 block decrement count 0 beta nil)
 (gd10 isa read-letters step start alpha 10 block decrement count 0 beta nil)

 (gi0 isa read-letters step start alpha 0 block increment count 0 beta nil)
 (gi1 isa read-letters step start alpha 1 block increment count 0 beta nil)
 (gi2 isa read-letters step start alpha 2 block increment count 0 beta nil)
 (gi3 isa read-letters step start alpha 3 block increment count 0 beta nil)
 (gi4 isa read-letters step start alpha 4 block increment count 0 beta nil)
 (gi5 isa read-letters step start alpha 5 block increment count 0 beta nil)
 (gi6 isa read-letters step start alpha 6 block increment count 0 beta nil)
 (gi7 isa read-letters step start alpha 7 block increment count 0 beta nil)
 (gi8 isa read-letters step start alpha 8 block increment count 0 beta nil)
 (gi9 isa read-letters step start alpha 9 block increment count 0 beta nil)
 (gi10 isa read-letters step start alpha 10 block increment count 0 beta nil)

 (gd0-b5 isa read-letters step start alpha 0 block decrement count 0 beta 5)
 (gd1-b5 isa read-letters step start alpha 1 block decrement count 0 beta 5)
 (gd2-b5 isa read-letters step start alpha 2 block decrement count 0 beta 5)
 (gd3-b5 isa read-letters step start alpha 3 block decrement count 0 beta 5)
 (gd4-b5 isa read-letters step start alpha 4 block decrement count 0 beta 5)
 (gd5-b5 isa read-letters step start alpha 5 block decrement count 0 beta 5)
 (gd6-b5 isa read-letters step start alpha 6 block decrement count 0 beta 5)
 (gd7-b5 isa read-letters step start alpha 7 block decrement count 0 beta 5)
 (gd8-b5 isa read-letters step start alpha 8 block decrement count 0 beta 5)
 (gd9-b5 isa read-letters step start alpha 9 block decrement count 0 beta 5)
 (gd10-b5 isa read-letters step start alpha 10 block decrement count 0 beta 5)

 (gi0-b5 isa read-letters step start alpha 0 block increment count 0 beta 5)
 (gi1-b5 isa read-letters step start alpha 1 block increment count 0 beta 5)
 (gi2-b5 isa read-letters step start alpha 2 block increment count 0 beta 5)
 (gi3-b5 isa read-letters step start alpha 3 block increment count 0 beta 5)
 (gi4-b5 isa read-letters step start alpha 4 block increment count 0 beta 5)
 (gi5-b5 isa read-letters step start alpha 5 block increment count 0 beta 5)
 (gi6-b5 isa read-letters step start alpha 6 block increment count 0 beta 5)
 (gi7-b5 isa read-letters step start alpha 7 block increment count 0 beta 5)
 (gi8-b5 isa read-letters step start alpha 8 block increment count 0 beta 5)
 (gi9-b5 isa read-letters step start alpha 9 block increment count 0 beta 5)
 (gi10-b5 isa read-letters step start alpha 10 block increment count 0 beta 5)

 (gd0-b6 isa read-letters step start alpha 0 block decrement count 0 beta 6)
 (gd1-b6 isa read-letters step start alpha 1 block decrement count 0 beta 6)
 (gd2-b6 isa read-letters step start alpha 2 block decrement count 0 beta 6)
 (gd3-b6 isa read-letters step start alpha 3 block decrement count 0 beta 6)
 (gd4-b6 isa read-letters step start alpha 4 block decrement count 0 beta 6)
 (gd5-b6 isa read-letters step start alpha 5 block decrement count 0 beta 6)
 (gd6-b6 isa read-letters step start alpha 6 block decrement count 0 beta 6)
 (gd7-b6 isa read-letters step start alpha 7 block decrement count 0 beta 6)
 (gd8-b6 isa read-letters step start alpha 8 block decrement count 0 beta 6)
 (gd9-b6 isa read-letters step start alpha 9 block decrement count 0 beta 6)
 (gd10-b6 isa read-letters step start alpha 10 block decrement count 0 beta 6)

 (gi0-b6 isa read-letters step start alpha 0 block increment count 0 beta 6)
 (gi1-b6 isa read-letters step start alpha 1 block increment count 0 beta 6)
 (gi2-b6 isa read-letters step start alpha 2 block increment count 0 beta 6)
 (gi3-b6 isa read-letters step start alpha 3 block increment count 0 beta 6)
 (gi4-b6 isa read-letters step start alpha 4 block increment count 0 beta 6)
 (gi5-b6 isa read-letters step start alpha 5 block increment count 0 beta 6)
 (gi6-b6 isa read-letters step start alpha 6 block increment count 0 beta 6)
 (gi7-b6 isa read-letters step start alpha 7 block increment count 0 beta 6)
 (gi8-b6 isa read-letters step start alpha 8 block increment count 0 beta 6)
 (gi9-b6 isa read-letters step start alpha 9 block increment count 0 beta 6)
 (gi10-b6 isa read-letters step start alpha 10 block increment count 0 beta 6)

 (gd0-b7 isa read-letters step start alpha 0 block decrement count 0 beta 7)
 (gd1-b7 isa read-letters step start alpha 1 block decrement count 0 beta 7)
 (gd2-b7 isa read-letters step start alpha 2 block decrement count 0 beta 7)
 (gd3-b7 isa read-letters step start alpha 3 block decrement count 0 beta 7)
 (gd4-b7 isa read-letters step start alpha 4 block decrement count 0 beta 7)
 (gd5-b7 isa read-letters step start alpha 5 block decrement count 0 beta 7)
 (gd6-b7 isa read-letters step start alpha 6 block decrement count 0 beta 7)
 (gd7-b7 isa read-letters step start alpha 7 block decrement count 0 beta 7)
 (gd8-b7 isa read-letters step start alpha 8 block decrement count 0 beta 7)
 (gd9-b7 isa read-letters step start alpha 9 block decrement count 0 beta 7)
 (gd10-b7 isa read-letters step start alpha 10 block decrement count 0 beta 7)

 (gi0-b7 isa read-letters step start alpha 0 block increment count 0 beta 7)
 (gi1-b7 isa read-letters step start alpha 1 block increment count 0 beta 7)
 (gi2-b7 isa read-letters step start alpha 2 block increment count 0 beta 7)
 (gi3-b7 isa read-letters step start alpha 3 block increment count 0 beta 7)
 (gi4-b7 isa read-letters step start alpha 4 block increment count 0 beta 7)
 (gi5-b7 isa read-letters step start alpha 5 block increment count 0 beta 7)
 (gi6-b7 isa read-letters step start alpha 6 block increment count 0 beta 7)
 (gi7-b7 isa read-letters step start alpha 7 block increment count 0 beta 7)
 (gi8-b7 isa read-letters step start alpha 8 block increment count 0 beta 7)
 (gi9-b7 isa read-letters step start alpha 9 block increment count 0 beta 7)
 (gi10-b7 isa read-letters step start alpha 10 block increment count 0 beta 7)

 (gd0-b8 isa read-letters step start alpha 0 block decrement count 0 beta 8)
 (gd1-b8 isa read-letters step start alpha 1 block decrement count 0 beta 8)
 (gd2-b8 isa read-letters step start alpha 2 block decrement count 0 beta 8)
 (gd3-b8 isa read-letters step start alpha 3 block decrement count 0 beta 8)
 (gd4-b8 isa read-letters step start alpha 4 block decrement count 0 beta 8)
 (gd5-b8 isa read-letters step start alpha 5 block decrement count 0 beta 8)
 (gd6-b8 isa read-letters step start alpha 6 block decrement count 0 beta 8)
 (gd7-b8 isa read-letters step start alpha 7 block decrement count 0 beta 8)
 (gd8-b8 isa read-letters step start alpha 8 block decrement count 0 beta 8)
 (gd9-b8 isa read-letters step start alpha 9 block decrement count 0 beta 8)
 (gd10-b8 isa read-letters step start alpha 10 block decrement count 0 beta 8)

 (gi0-b8 isa read-letters step start alpha 0 block increment count 0 beta 8)
 (gi1-b8 isa read-letters step start alpha 1 block increment count 0 beta 8)
 (gi2-b8 isa read-letters step start alpha 2 block increment count 0 beta 8)
 (gi3-b8 isa read-letters step start alpha 3 block increment count 0 beta 8)
 (gi4-b8 isa read-letters step start alpha 4 block increment count 0 beta 8)
 (gi5-b8 isa read-letters step start alpha 5 block increment count 0 beta 8)
 (gi6-b8 isa read-letters step start alpha 6 block increment count 0 beta 8)
 (gi7-b8 isa read-letters step start alpha 7 block increment count 0 beta 8)
 (gi8-b8 isa read-letters step start alpha 8 block increment count 0 beta 8)
 (gi9-b8 isa read-letters step start alpha 9 block increment count 0 beta 8)
 (gi10-b8 isa read-letters step start alpha 10 block increment count 0 beta 8)

 (gd0-b9 isa read-letters step start alpha 0 block decrement count 0 beta 9)
 (gd1-b9 isa read-letters step start alpha 1 block decrement count 0 beta 9)
 (gd2-b9 isa read-letters step start alpha 2 block decrement count 0 beta 9)
 (gd3-b9 isa read-letters step start alpha 3 block decrement count 0 beta 9)
 (gd4-b9 isa read-letters step start alpha 4 block decrement count 0 beta 9)
 (gd5-b9 isa read-letters step start alpha 5 block decrement count 0 beta 9)
 (gd6-b9 isa read-letters step start alpha 6 block decrement count 0 beta 9)
 (gd7-b9 isa read-letters step start alpha 7 block decrement count 0 beta 9)
 (gd8-b9 isa read-letters step start alpha 8 block decrement count 0 beta 9)
 (gd9-b9 isa read-letters step start alpha 9 block decrement count 0 beta 9)
 (gd10-b9 isa read-letters step start alpha 10 block decrement count 0 beta 9)

 (gi0-b9 isa read-letters step start alpha 0 block increment count 0 beta 9)
 (gi1-b9 isa read-letters step start alpha 1 block increment count 0 beta 9)
 (gi2-b9 isa read-letters step start alpha 2 block increment count 0 beta 9)
 (gi3-b9 isa read-letters step start alpha 3 block increment count 0 beta 9)
 (gi4-b9 isa read-letters step start alpha 4 block increment count 0 beta 9)
 (gi5-b9 isa read-letters step start alpha 5 block increment count 0 beta 9)
 (gi6-b9 isa read-letters step start alpha 6 block increment count 0 beta 9)
 (gi7-b9 isa read-letters step start alpha 7 block increment count 0 beta 9)
 (gi8-b9 isa read-letters step start alpha 8 block increment count 0 beta 9)
 (gi9-b9 isa read-letters step start alpha 9 block increment count 0 beta 9)
 (gi10-b9 isa read-letters step start alpha 10 block increment count 0 beta 9)

 (gd0-b10 isa read-letters step start alpha 0 block decrement count 0 beta 10)
 (gd1-b10 isa read-letters step start alpha 1 block decrement count 0 beta 10)
 (gd2-b10 isa read-letters step start alpha 2 block decrement count 0 beta 10)
 (gd3-b10 isa read-letters step start alpha 3 block decrement count 0 beta 10)
 (gd4-b10 isa read-letters step start alpha 4 block decrement count 0 beta 10)
 (gd5-b10 isa read-letters step start alpha 5 block decrement count 0 beta 10)
 (gd6-b10 isa read-letters step start alpha 6 block decrement count 0 beta 10)
 (gd7-b10 isa read-letters step start alpha 7 block decrement count 0 beta 10)
 (gd8-b10 isa read-letters step start alpha 8 block decrement count 0 beta 10)
 (gd9-b10 isa read-letters step start alpha 9 block decrement count 0 beta 10)
 (gd10-b10 isa read-letters step start alpha 10 block decrement count 0 beta 10)

 (gi0-b10 isa read-letters step start alpha 0 block increment count 0 beta 10)
 (gi1-b10 isa read-letters step start alpha 1 block increment count 0 beta 10)
 (gi2-b10 isa read-letters step start alpha 2 block increment count 0 beta 10)
 (gi3-b10 isa read-letters step start alpha 3 block increment count 0 beta 10)
 (gi4-b10 isa read-letters step start alpha 4 block increment count 0 beta 10)
 (gi5-b10 isa read-letters step start alpha 5 block increment count 0 beta 10)
 (gi6-b10 isa read-letters step start alpha 6 block increment count 0 beta 10)
 (gi7-b10 isa read-letters step start alpha 7 block increment count 0 beta 10)
 (gi8-b10 isa read-letters step start alpha 8 block increment count 0 beta 10)
 (gi9-b10 isa read-letters step start alpha 9 block increment count 0 beta 10)
 (gi10-b10 isa read-letters step start alpha 10 block increment count 0 beta 10)

 (gd0-b11 isa read-letters step start alpha 0 block decrement count 0 beta 11)
 (gd1-b11 isa read-letters step start alpha 1 block decrement count 0 beta 11)
 (gd2-b11 isa read-letters step start alpha 2 block decrement count 0 beta 11)
 (gd3-b11 isa read-letters step start alpha 3 block decrement count 0 beta 11)
 (gd4-b11 isa read-letters step start alpha 4 block decrement count 0 beta 11)
 (gd5-b11 isa read-letters step start alpha 5 block decrement count 0 beta 11)
 (gd6-b11 isa read-letters step start alpha 6 block decrement count 0 beta 11)
 (gd7-b11 isa read-letters step start alpha 7 block decrement count 0 beta 11)
 (gd8-b11 isa read-letters step start alpha 8 block decrement count 0 beta 11)
 (gd9-b11 isa read-letters step start alpha 9 block decrement count 0 beta 11)
 (gd10-b11 isa read-letters step start alpha 10 block decrement count 0 beta 11)

 (gi0-b11 isa read-letters step start alpha 0 block increment count 0 beta 11)
 (gi1-b11 isa read-letters step start alpha 1 block increment count 0 beta 11)
 (gi2-b11 isa read-letters step start alpha 2 block increment count 0 beta 11)
 (gi3-b11 isa read-letters step start alpha 3 block increment count 0 beta 11)
 (gi4-b11 isa read-letters step start alpha 4 block increment count 0 beta 11)
 (gi5-b11 isa read-letters step start alpha 5 block increment count 0 beta 11)
 (gi6-b11 isa read-letters step start alpha 6 block increment count 0 beta 11)
 (gi7-b11 isa read-letters step start alpha 7 block increment count 0 beta 11)
 (gi8-b11 isa read-letters step start alpha 8 block increment count 0 beta 11)
 (gi9-b11 isa read-letters step start alpha 9 block increment count 0 beta 11)
 (gi10-b11 isa read-letters step start alpha 10 block increment count 0 beta 11)

 (gd0-b12 isa read-letters step start alpha 0 block decrement count 0 beta 12)
 (gd1-b12 isa read-letters step start alpha 1 block decrement count 0 beta 12)
 (gd2-b12 isa read-letters step start alpha 2 block decrement count 0 beta 12)
 (gd3-b12 isa read-letters step start alpha 3 block decrement count 0 beta 12)
 (gd4-b12 isa read-letters step start alpha 4 block decrement count 0 beta 12)
 (gd5-b12 isa read-letters step start alpha 5 block decrement count 0 beta 12)
 (gd6-b12 isa read-letters step start alpha 6 block decrement count 0 beta 12)
 (gd7-b12 isa read-letters step start alpha 7 block decrement count 0 beta 12)
 (gd8-b12 isa read-letters step start alpha 8 block decrement count 0 beta 12)
 (gd9-b12 isa read-letters step start alpha 9 block decrement count 0 beta 12)
 (gd10-b12 isa read-letters step start alpha 10 block decrement count 0 beta 12)

 (gi0-b12 isa read-letters step start alpha 0 block increment count 0 beta 12)
 (gi1-b12 isa read-letters step start alpha 1 block increment count 0 beta 12)
 (gi2-b12 isa read-letters step start alpha 2 block increment count 0 beta 12)
 (gi3-b12 isa read-letters step start alpha 3 block increment count 0 beta 12)
 (gi4-b12 isa read-letters step start alpha 4 block increment count 0 beta 12)
 (gi5-b12 isa read-letters step start alpha 5 block increment count 0 beta 12)
 (gi6-b12 isa read-letters step start alpha 6 block increment count 0 beta 12)
 (gi7-b12 isa read-letters step start alpha 7 block increment count 0 beta 12)
 (gi8-b12 isa read-letters step start alpha 8 block increment count 0 beta 12)
 (gi9-b12 isa read-letters step start alpha 9 block increment count 0 beta 12)
 (gi10-b12 isa read-letters step start alpha 10 block increment count 0 beta 12)

 (gd0-b13 isa read-letters step start alpha 0 block decrement count 0 beta 13)
 (gd1-b13 isa read-letters step start alpha 1 block decrement count 0 beta 13)
 (gd2-b13 isa read-letters step start alpha 2 block decrement count 0 beta 13)
 (gd3-b13 isa read-letters step start alpha 3 block decrement count 0 beta 13)
 (gd4-b13 isa read-letters step start alpha 4 block decrement count 0 beta 13)
 (gd5-b13 isa read-letters step start alpha 5 block decrement count 0 beta 13)
 (gd6-b13 isa read-letters step start alpha 6 block decrement count 0 beta 13)
 (gd7-b13 isa read-letters step start alpha 7 block decrement count 0 beta 13)
 (gd8-b13 isa read-letters step start alpha 8 block decrement count 0 beta 13)
 (gd9-b13 isa read-letters step start alpha 9 block decrement count 0 beta 13)
 (gd10-b13 isa read-letters step start alpha 10 block decrement count 0 beta 13)

 (gi0-b13 isa read-letters step start alpha 0 block increment count 0 beta 13)
 (gi1-b13 isa read-letters step start alpha 1 block increment count 0 beta 13)
 (gi2-b13 isa read-letters step start alpha 2 block increment count 0 beta 13)
 (gi3-b13 isa read-letters step start alpha 3 block increment count 0 beta 13)
 (gi4-b13 isa read-letters step start alpha 4 block increment count 0 beta 13)
 (gi5-b13 isa read-letters step start alpha 5 block increment count 0 beta 13)
 (gi6-b13 isa read-letters step start alpha 6 block increment count 0 beta 13)
 (gi7-b13 isa read-letters step start alpha 7 block increment count 0 beta 13)
 (gi8-b13 isa read-letters step start alpha 8 block increment count 0 beta 13)
 (gi9-b13 isa read-letters step start alpha 9 block increment count 0 beta 13)
 (gi10-b13 isa read-letters step start alpha 10 block increment count 0 beta 13)

 (gd0-b14 isa read-letters step start alpha 0 block decrement count 0 beta 14)
 (gd1-b14 isa read-letters step start alpha 1 block decrement count 0 beta 14)
 (gd2-b14 isa read-letters step start alpha 2 block decrement count 0 beta 14)
 (gd3-b14 isa read-letters step start alpha 3 block decrement count 0 beta 14)
 (gd4-b14 isa read-letters step start alpha 4 block decrement count 0 beta 14)
 (gd5-b14 isa read-letters step start alpha 5 block decrement count 0 beta 14)
 (gd6-b14 isa read-letters step start alpha 6 block decrement count 0 beta 14)
 (gd7-b14 isa read-letters step start alpha 7 block decrement count 0 beta 14)
 (gd8-b14 isa read-letters step start alpha 8 block decrement count 0 beta 14)
 (gd9-b14 isa read-letters step start alpha 9 block decrement count 0 beta 14)
 (gd10-b14 isa read-letters step start alpha 10 block decrement count 0 beta 14)

 (gi0-b14 isa read-letters step start alpha 0 block increment count 0 beta 14)
 (gi1-b14 isa read-letters step start alpha 1 block increment count 0 beta 14)
 (gi2-b14 isa read-letters step start alpha 2 block increment count 0 beta 14)
 (gi3-b14 isa read-letters step start alpha 3 block increment count 0 beta 14)
 (gi4-b14 isa read-letters step start alpha 4 block increment count 0 beta 14)
 (gi5-b14 isa read-letters step start alpha 5 block increment count 0 beta 14)
 (gi6-b14 isa read-letters step start alpha 6 block increment count 0 beta 14)
 (gi7-b14 isa read-letters step start alpha 7 block increment count 0 beta 14)
 (gi8-b14 isa read-letters step start alpha 8 block increment count 0 beta 14)
 (gi9-b14 isa read-letters step start alpha 9 block increment count 0 beta 14)
 (gi10-b14 isa read-letters step start alpha 10 block increment count 0 beta 14)

 (gd0-b15 isa read-letters step start alpha 0 block decrement count 0 beta 15)
 (gd1-b15 isa read-letters step start alpha 1 block decrement count 0 beta 15)
 (gd2-b15 isa read-letters step start alpha 2 block decrement count 0 beta 15)
 (gd3-b15 isa read-letters step start alpha 3 block decrement count 0 beta 15)
 (gd4-b15 isa read-letters step start alpha 4 block decrement count 0 beta 15)
 (gd5-b15 isa read-letters step start alpha 5 block decrement count 0 beta 15)
 (gd6-b15 isa read-letters step start alpha 6 block decrement count 0 beta 15)
 (gd7-b15 isa read-letters step start alpha 7 block decrement count 0 beta 15)
 (gd8-b15 isa read-letters step start alpha 8 block decrement count 0 beta 15)
 (gd9-b15 isa read-letters step start alpha 9 block decrement count 0 beta 15)
 (gd10-b15 isa read-letters step start alpha 10 block decrement count 0 beta 15)

 (gi0-b15 isa read-letters step start alpha 0 block increment count 0 beta 15)
 (gi1-b15 isa read-letters step start alpha 1 block increment count 0 beta 15)
 (gi2-b15 isa read-letters step start alpha 2 block increment count 0 beta 15)
 (gi3-b15 isa read-letters step start alpha 3 block increment count 0 beta 15)
 (gi4-b15 isa read-letters step start alpha 4 block increment count 0 beta 15)
 (gi5-b15 isa read-letters step start alpha 5 block increment count 0 beta 15)
 (gi6-b15 isa read-letters step start alpha 6 block increment count 0 beta 15)
 (gi7-b15 isa read-letters step start alpha 7 block increment count 0 beta 15)
 (gi8-b15 isa read-letters step start alpha 8 block increment count 0 beta 15)
 (gi9-b15 isa read-letters step start alpha 9 block increment count 0 beta 15)
 (gi10-b15 isa read-letters step start alpha 10 block increment count 0 beta 15)

 (gd0-b16 isa read-letters step start alpha 0 block decrement count 0 beta 16)
 (gd1-b16 isa read-letters step start alpha 1 block decrement count 0 beta 16)
 (gd2-b16 isa read-letters step start alpha 2 block decrement count 0 beta 16)
 (gd3-b16 isa read-letters step start alpha 3 block decrement count 0 beta 16)
 (gd4-b16 isa read-letters step start alpha 4 block decrement count 0 beta 16)
 (gd5-b16 isa read-letters step start alpha 5 block decrement count 0 beta 16)
 (gd6-b16 isa read-letters step start alpha 6 block decrement count 0 beta 16)
 (gd7-b16 isa read-letters step start alpha 7 block decrement count 0 beta 16)
 (gd8-b16 isa read-letters step start alpha 8 block decrement count 0 beta 16)
 (gd9-b16 isa read-letters step start alpha 9 block decrement count 0 beta 16)
 (gd10-b16 isa read-letters step start alpha 10 block decrement count 0 beta 16)

 (gi0-b16 isa read-letters step start alpha 0 block increment count 0 beta 16)
 (gi1-b16 isa read-letters step start alpha 1 block increment count 0 beta 16)
 (gi2-b16 isa read-letters step start alpha 2 block increment count 0 beta 16)
 (gi3-b16 isa read-letters step start alpha 3 block increment count 0 beta 16)
 (gi4-b16 isa read-letters step start alpha 4 block increment count 0 beta 16)
 (gi5-b16 isa read-letters step start alpha 5 block increment count 0 beta 16)
 (gi6-b16 isa read-letters step start alpha 6 block increment count 0 beta 16)
 (gi7-b16 isa read-letters step start alpha 7 block increment count 0 beta 16)
 (gi8-b16 isa read-letters step start alpha 8 block increment count 0 beta 16)
 (gi9-b16 isa read-letters step start alpha 9 block increment count 0 beta 16)
 (gi10-b16 isa read-letters step start alpha 10 block increment count 0 beta 16)
)

(P find-unattended-letter
   =goal>
      ISA         read-letters
      step        start
   ?retrieval>
      state       free
 ==>
   +visual-location>
      :attended   nil
   =goal>
      step        find-location
   +retrieval>
      isa         stimulus
      type        standard
)

(P attend-letter
   =goal>
      ISA         read-letters
      step        find-location
   =visual-location>
   ?visual>
      state       free
==>
   +visual>
      cmd         move-attention
      screen-pos  =visual-location
   =goal>
      step        attend
   +temporal>
      cmd         time
      ticks       0
)

(P encode-duration
   =goal>
      ISA         read-letters
      step        attend
   =visual>
      value       =letter
==>
   =goal>
      step        wait
)

(P encode-finish-compare
   =goal>
      step        wait
   ?visual>
      buffer      unrequested
      state       error
   =temporal>
      ticks       =ticks
   ?imaginal>
      state       free
==>
   =goal>
      step       compare
   +temporal>
      cmd        clear
   +imaginal>
      isa        stimulus
      duration   =ticks
)

(P encode-finish-recall
   =goal>
      step        wait
   ?visual>
      buffer      unrequested
      state       error
   =temporal>
      ticks       =ticks
   ?imaginal>
      state       free
   ?retrieval>
      state       free
==>
   =goal>
      step       recall
   +temporal>
      cmd        clear
   +imaginal>
      isa        stimulus
      duration   =ticks
   +retrieval>
      isa        stimulus
      duration   =ticks
    - type       nil
)

(P encode-finish-guess
   =goal>
      step        wait
      count       =b
   -  beta        nil
      beta        =b
   ?visual>
      buffer      unrequested
      state       error
   =temporal>
      ticks       =ticks
   ?imaginal>
      state       free
==>
   =goal>
      step       respond
      confident  no
   +temporal>
      cmd        clear
   +imaginal>
      isa        stimulus
      duration   =ticks
      type       oddball
)

(P compare-no-target
   =goal>
      step       compare
   ?retrieval>
      state      error
   =imaginal>
      isa        stimulus
      duration   =ticks
   ?imaginal>
      state      free
==>
   =goal>
      step       start
   =imaginal>
      type       standard
   -imaginal>
)

(P compare-match-decrement
   !bind! =ticks-minus-threshold (- =ticks =threshold)
   =goal>
      step       compare
      alpha      =threshold
      block      decrement
   =retrieval>
      duration   =ticks
   =imaginal>
      isa        stimulus
   >= duration   =ticks-minus-threshold
   ?imaginal>
      state      free
==>
   =goal>
      step       start
   =imaginal>
      type       standard
   -imaginal>
)

(P compare-match-increment
   !bind! =ticks-plus-threshold (+ =ticks =threshold)
   =goal>
      step       compare
      alpha      =threshold
      block      increment
   =retrieval>
      duration   =ticks
   =imaginal>
      isa        stimulus
   <= duration   =ticks-plus-threshold
   ?imaginal>
      state      free
==>
   =goal>
      step       start
   =imaginal>
      type       standard
   -imaginal>
)

(P compare-mismatch-shorter
   !bind! =ticks-minus-threshold (- =ticks =threshold)
   =goal>
      step       compare
      alpha      =threshold
      block      decrement
   =retrieval>
      duration   =ticks
   =imaginal>
      isa        stimulus
   <  duration   =ticks-minus-threshold
   ?imaginal>
      state      free
==>
   =goal>
      step       respond
      confident  yes
   =imaginal>
      type       oddball
   -imaginal>
)

(P compare-mismatch-longer
   !bind! =ticks-plus-threshold (+ =ticks =threshold)
   =goal>
      step       compare
      alpha      =threshold
      block      increment
   =retrieval>
      duration   =ticks
   =imaginal>
      isa        stimulus
   >  duration   =ticks-plus-threshold
   ?imaginal>
      state      free
==>
   =goal>
      step       respond
      confident  yes
   =imaginal>
      type       oddball
   -imaginal>
)

(P recall-standard
   !bind! =i-update (+ =i 1)
   =goal>
      step       recall
      count      =i
   =retrieval>
      duration   =ticks
      type       standard
   ?imaginal>
      state      free
   =imaginal>
==>
   =goal>
      step       start
      count      =i-update
   =imaginal>
      type       standard
   -imaginal>
)

(P recall-oddball
   =goal>
      step       recall
   =retrieval>
      duration   =ticks
      type       oddball
   ?imaginal>
      state      free
   =imaginal>
==>
   =goal>
      step       respond
      confident  yes
   =imaginal>
      type       oddball
   -imaginal>
)

(P recall-error-respond
   =goal>
      step       recall
   ?retrieval>
      state      error
   ?imaginal>
      state      free
   =imaginal>
==>
   =goal>
      step       respond
      confident  no
   =imaginal>
      type       oddball
   -imaginal>
)

(P recall-error-not-respond
   !bind! =i-update (+ =i 1)
   =goal>
      step       recall
      count      =i
   ?retrieval>
      state      error
   ?imaginal>
      state      free
   =imaginal>
==>
   =goal>
      step       start
      count      =i-update
   =imaginal>
      type       standard
   -imaginal>
)

(P respond-confident
   =goal>
      ISA         read-letters
      step        respond
      confident   yes
   ?manual>   
      state       free
==>
   =goal>
      step        start
      count       0
   +manual>
      cmd         press-key
      key         j
)

(P respond-unconfident
   =goal>
      ISA         read-letters
      step        respond
      confident   no
   ?manual>
      state       free
==>
   =goal>
      step        start
      count       0
   +manual>
      cmd         press-key
      key         j
)

(spp encode-finish-compare :u 65)
(spp encode-finish-recall :u 0)

(spp respond-confident :reward 1)
(spp respond-unconfident :reward -1)

)
