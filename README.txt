Project with a set of images for training a neural network to flatten
(and enhance) the images.

  "Ideal" flattened pages are located at:
    blend/ideal/
  Warped/bent page images are located at:
    blend/pages/
  Source words (not used except to generate images) are found in:
    blend/words.txt
    * Note: Not all lines were used to process the images in these
            initial tests.

Naming convention:

A checksum of the text used to create the image sets are used as unique
identifiers, this allows us to relate the specific bent and flattened images to
each other.

For example, the random words, "Reva's unmask Afro freaky" (separated by
newlines) ends up being "4b8ba00fb963994e".

A single flattened version of a page with the text is then stored at:
  ./blend/ideal/4b8ba00fb963994e/0001.png
  (In the future I might create many ideal versions for training)

Then many bent versions of the page are stored as:
  ./blend/pages/4b8ba00fb963994e/0000.png
  ./blend/pages/4b8ba00fb963994e/0001.png
  ...
  ./blend/pages/4b8ba00fb963994e/0099.png

*Important note:
  The sequential numbering within the folders expresses no special
  relationship.  Thus, pages/.../0001.png is just one version of a
  bent page, just like 0002.png, and the ideal/.../0001.png is a flattened
  version of both 0001.jpg and 0002.jpg (and all the others in that
  corresponding pages/.../ folder).

Questions go to: jaggz.h at g mail com
Or: jaggz@freenode
