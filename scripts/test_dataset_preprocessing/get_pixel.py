"""
MIT License

Copyright (c) 2025 Nate Gillman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np  
import cv2    
print(cv2.__version__)


def show_pixel(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get image height to convert y-coordinate (invert y to use bottom-left as origin)
        height = img.shape[0]
        # The y-coordinate is inverted: y_bottom_left = height - 1 - y_top_left
        y_bottom_left = height - 1 - y
        print(f"x,y = {x},{y_bottom_left}") # Original y for pixel lookup, converted y for display
  
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='pick pixel value in an image!')
    parser.add_argument("imagename",
                        metavar="imagename",
                        help="'path to image and name")
    args = parser.parse_args()
    print("path: ",args.imagename)


    #img = np.zeros((512, 512, 3), np.uint8)
    img = cv2.imread(args.imagename)
    
    # Get and print image dimensions
    height, width = img.shape[:2]
    print(f"Image dimensions: width={width}, height={height}")
    
    cv2.namedWindow('image')      
    cv2.setMouseCallback('image', show_pixel)    
            
    while (1):
       cv2.imshow('image', img)
       if cv2.waitKey(2) & 0xFF == 27:       
           break  
    cv2.destroyAllWindows()