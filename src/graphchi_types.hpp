
/*
 Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef DEF_GRAPHCHI_TYPES
#define DEF_GRAPHCHI_TYPES


#include <stdint.h>
#include <iostream>

namespace graphchi {
    
    typedef uint32_t vid_t;
    
    
    /** 
      * PairContainer encapsulates a pair of values of some type.
      * Useful for bulk-synchronuos computation.
      */
    template <typename ET>
    struct PairContainer {
        ET left;
        ET right;
        
        PairContainer() {
            left = ET();
            right = ET();
        }
        
        PairContainer(ET a, ET b) {
            left = a;
            right = b;
        }
        
        ET & oldval(int iter) {
            return (iter % 2 == 0 ? left : right);
        }
        
        void set_newval(int iter, ET x) {
            if (iter % 2 == 0) {
                right = x;
            } else {
                left = x;
            }
        }

#ifdef DYNAMICEDATA
        typedef typename ET::element_type_t element_type_t;
        typedef uint32_t sizeword_t;


        PairContainer(uint16_t sz, uint16_t cap, element_type_t * dataptr) {
            int lsz = dataptr[0];
            int lcsz = dataptr[1];
            dataptr += 2;

            //std::cout << "size: " << sizeof(element_type_t) << std::endl;
            //std::cout << "lsz: " << lsz << ", rsz: " << (sz - lsz) << std::endl;
            //std::cout << "lcsz: " << lcsz << ", rcsz: " << (cap - lcsz) << std::endl;


            left = ET(lsz, lcsz, dataptr);
            dataptr += lsz;
            right = ET(sz - lsz, cap - lcsz, dataptr);
        }


        void write(element_type_t * dest){
            int lsz = left.size();
            int rsz = right.size();
            dest[0] = lsz;
            dest[1] = left.capacity();
            for(int i = 0; i < lsz; i++){
                std::cout << "left: " << left[i] << std::endl;
                dest[i + 2] = left[i];
            }
            for(int i = 0; i < rsz; i++)
                dest[lsz + i + 2] = right[i];
        }

        uint16_t size(){
            return left.size() + right.size() + 2;
        }

        uint16_t capacity(){
            return left.capacity() + right.capacity();
        }
#endif
    };
    
    struct shard_index {
        vid_t vertexid;
        size_t filepos;
        size_t edgecounter;
        shard_index(vid_t vertexid, size_t filepos, size_t edgecounter) : vertexid(vertexid), filepos(filepos), edgecounter(edgecounter) {}
    };
    

}


#endif



