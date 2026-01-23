chunk0: [  0  1  2  3 ]
chunk1: [  4  5  6  7 ]
chunk2: [  8  9 10 11 ]
chunk3: [ 12 13 14 15 ]
chunk4: [ 16 17 18 19 ]
chunk5: [ 20 21 22 23 ]
chunk6: [ 24 25 26 27 ]
chunk7: [ 28 29 30 31 ]

window_size = 16
memory_size = 20
full latent = 32

------------------------------------------------------------
in window (20%)
Always use full 16 latent (4 chunk) do bidirectional noise cancel
[  0  1  2  3 ][  4  5  6  7 ][  8  9 10 11 ][ 12 13 14 15 ]

------------------------------------------------------------

out window
First select a number after window size, smaller than full latent: {16,20,24}

[  0  1  2  3 ][  4  5  6  7 ][  8  9 10 11 ][ 12 13 14 15 ][ 16 17 18 19 ][ 20 21 22 23 ][ 24 25 26 27 ][ 28 29 30 31 ]
  chunk0          chunk1          chunk2          chunk3          chunk4          chunk5       chunk6         chunk7

------------------------------------------------------------

For example, selecting to train 24 (chunk6)
context_latent = array(size=memory_size=20)
First add dding the first chunk0 to context_latent
Second add temporal 12 latent (chunk345) into context_latent

Then, use FOV overlapping, to select the maximum overlapped chunk (for example chunk2)

Finally, the context memory look like this:
[  0  1  2  3 ] + [  8  9 10 11 ] + [ 12 13 14 15 ][ 16 17 18 19 ][ 20 21 22 23 ] + [ 24 25 26 27 ]
  chunk0           chunk2            chunk3          chunk4          chunk5           chunk6

only train on training chunk6 noise cancal