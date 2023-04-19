This implementation is based off of [this paper by FAIR](https://arxiv.org/pdf/2104.14294.pdf).

In a nutshell, the paper attempts to take two different augmentations of the same image, and try and 
push these embeddings closer together. Most other tasks attempt to do a triplet loss where the image is compared to 
a similar image (positive example), and different image(s) (negative examples). What's amazing about this paper is 
that it ignores negative examples altogether and is still able to get a meaningful embedding space for a given image.

