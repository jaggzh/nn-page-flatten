Project with a set of images for training a neural network to flatten
(and enhance) the images.

  Ideal flattened pages are located at:
    blend/ideal/
  Warped page images are located at:
    blend/pages/
  Source words (not used except to generate images) are found in:
    blend/words.txt
    * Note: Not all lines were used to process the images in these
            initial tests.

Naming convention:

A checksum of the text used to create the image sets are used as unique
identifiers, this allows us to relate the specific bent and curved images to
each other.

For example, the random words, "Reva's unmask Afro freaky" (separated by
newlines) ends up being "4b8ba00fb963994e".

Thus, we have a set of curved images, like:
  ./blend/pages/4b8ba00fb963994e/0000.png
  ./blend/pages/4b8ba00fb963994e/0001.png
  ...
  ./blend/pages/4b8ba00fb963994e/0099.png

The ideal flattened versions are found as:
  ./blend/ideal/4b8ba00fb963994e/0001.png

I currently generate just a single ideal image.

*Important note:
  ALL the images in the pages/4b8ba00fb963994e/ folder relate to the
   flattened ideal/4b8ba00fb963994e/ image(s);
  the point being that THE NUMBERING OF THE IMAGES IS NOT RELEVANT AT ALL,
  so, pages/whatever/0001.jpg, 0002.jpg, etc. are all just bent versions
  of ideal/whatever/0001.jpg, and the 0001.jpg in either folder does not
  reflect a special relationship between those numbers.

Questions go to: jaggz.h at g mail com
