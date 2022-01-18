# TexturalFilter
This is the program used by the UdeS to parcipate in IBSI chapter 2.
All test images are available in the Data repertory.
The result of the UdeS teams and the McGill teams are available in the Result_Alex and Result_Martin repertory.

## GPU usage
To accelerate the computing with the mean, laplacian of gaussian, laws, or gabor filter you can use your GPU if it's compatible with pytorch.
Otherwise, all filter maps can be compute on a cpu.

## Exemple of usage
```
python main.py --test_id=1a1 --device=cuda:0 --compare
```
