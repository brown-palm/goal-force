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

def split_list_across_devices_contiguous(items, world_size, device_id):
    ####################################################################################
    ################ Logic for splitting computation across devices
    ####################################################################################
    
    """
    Split a list of items into contiguous chunks across devices.
    
    Example: [a, b, c, d, e] with world_size=2
    - device 0 gets [a, b, c]
    - device 1 gets [d, e]
    """
    n = len(items)
    
    # Calculate the base size for each chunk and the remainder
    base_size = n // world_size
    remainder = n % world_size
    # The first 'remainder' devices get one extra item (base_size + 1)
    # The remaining 'world_size - remainder' devices get 'base_size' items
    if device_id < remainder:
        # This device gets the larger chunk size
        chunk_size = base_size + 1
        # The start index is simply its ID times the larger chunk size
        start_index = device_id * chunk_size
    else:
        # This device gets the base chunk size
        chunk_size = base_size
        # The start index is offset by all the 'remainder' larger chunks
        # that came before it, plus its relative position among the smaller chunks.
        start_index = remainder * (base_size + 1) + (device_id - remainder) * base_size
    end_index = start_index + chunk_size
    # Return the slice for this device
    return items[start_index:end_index]
