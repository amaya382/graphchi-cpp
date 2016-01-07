/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 
 *
 * @section DESCRIPTION
 *
 * Bulk-synchronous implementation of the functional API.
 * This API can be used to implement Sparse-Matrix-Vector-Multiply programs.
 *
 * @section TODO
 *
 * There is too much common code with the semi-sync version. Consolidate!
 */



#ifndef GRAPHCHI_FUNCTIONAL_BULKSYNC_DEF
#define GRAPHCHI_FUNCTIONAL_BULKSYNC_DEF

#include <assert.h>
#include <immintrin.h>



#include "api/graph_objects.hpp"
#include "api/graphchi_context.hpp"
#include "api/functional/functional_defs.hpp"

#include "metrics/metrics.hpp"
#include "graphchi_types.hpp"

namespace graphchi {
    
    template <typename KERNEL>
    class functional_vertex_unweighted_bulksync : public graphchi_vertex<typename KERNEL::VertexDataType, PairContainer<typename KERNEL::EdgeDataType>> {
    public:

        typedef typename KERNEL::EdgeDataType ET;
        typedef typename KERNEL::VertexDataType VT;
        typedef PairContainer<typename KERNEL::EdgeDataType> P_ET;

        KERNEL kernel;

        ET acc;
        
        vertex_info vinfo;
        graphchi_context * gcontext;
        
        functional_vertex_unweighted_bulksync() : graphchi_vertex<VT, P_ET> () {}
        
        functional_vertex_unweighted_bulksync(graphchi_context &ginfo, vid_t _id, int indeg, int outdeg) : 
        graphchi_vertex<VT, P_ET> (_id, NULL, NULL, indeg, outdeg) {
            vinfo.indegree = indeg;
            vinfo.outdegree = outdeg;
            vinfo.vertexid = _id;
            acc = kernel.zero();
            gcontext = &ginfo;
        }

        vid_t id() const {
            return vinfo.vertexid;
        }
        
        functional_vertex_unweighted_bulksync(vid_t _id, 
                                              graphchi_edge<P_ET> * iptr,
                                              graphchi_edge<P_ET> * optr,
                                              int indeg, 
                                              int outdeg) {
            assert(false); // This should never be called.
        }
        
        void first_iteration(graphchi_context &ginfo) {
            this->set_data(kernel.init(ginfo, vinfo));
            gcontext = &ginfo;
        }
        
        // Optimization: as only memshard (not streaming shard) creates inedgers,
        // we do not need atomic instructions here!
        inline void add_inedge(vid_t src, P_ET * ptr, bool special_edge) {
            if (gcontext->iteration > 0) {
                auto val = kernel.gather(*gcontext,
                                                 vinfo,
                                                 src,
                                                 ptr->oldval(gcontext->iteration));
#ifdef HTM
#if !defined(RTM) && !defined(HLE)
                __transaction_relaxed
#else
#ifdef RTM
                retry:
                int st = _xbegin();
                if (st == _XBEGIN_STARTED) {
#else
#ifdef HLE
                int lock;
                while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE | __ATOMIC_HLE_ACQUIRE))
                    _mm_pause();
#endif
#endif
#endif
#else
                get_lock(vinfo.vertexid).lock();
#endif
                {
                    acc = kernel.plus(acc, val);
                }
#ifdef HTM
#ifdef RTM
                    _xend();
                } else {
                    goto retry;
                }
#else
#ifdef HLE
                __atomic_clear(&lock, __ATOMIC_RELEASE | __ATOMIC_HLE_RELEASE);
#endif
#endif
#else
                get_lock(vinfo.vertexid).unlock();
#endif
            }
        }
        
        void ready(graphchi_context &ginfo) {
            this->set_data(kernel.apply(*gcontext, vinfo, acc));
        }
        
        inline void add_outedge(vid_t dst, P_ET * ptr, bool special_edge) {
            typename KERNEL::EdgeDataType newval = 
                kernel.scatter(*gcontext, vinfo, dst, this->get_data());
            P_ET paircont = *ptr;
            paircont.set_newval(gcontext->iteration, newval);
            *ptr = paircont;
        }
        
        bool computational_edges() {
            return true;
        }
        
        /** 
          * We also need to read the outedges, because we need
          * to preserve the old value as well.
          */
        static bool read_outedges() {
            return true;
        }
    };
    
    
    
    template <typename KERNEL>
    class FunctionalProgramProxyBulkSync : public GraphChiProgram<typename KERNEL::VertexDataType,  PairContainer<typename KERNEL::EdgeDataType>, functional_vertex_unweighted_bulksync<KERNEL>  > {
    public:  
   
        typedef typename KERNEL::VertexDataType VertexDataType;
        typedef PairContainer<typename KERNEL::EdgeDataType> EdgeDataType;
        typedef functional_vertex_unweighted_bulksync<KERNEL> fvertex_t;
        
        /**
         * Called before an iteration starts.
         */
        void before_iteration(int iteration, graphchi_context &info) {
        }
        
        /**
         * Called after an iteration has finished.
         */
        void after_iteration(int iteration, graphchi_context &ginfo) {
        }
        
        /**
         * Called before an execution interval is started.
         */
        void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {        
        }
        
        
        /**
         * Pagerank update function.
         */
        void update(fvertex_t &v, graphchi_context &ginfo) {
            if (ginfo.iteration == 0) {
                v.first_iteration(ginfo);
            } else { 
                v.ready(ginfo);
            }   
            
        }
        
    };
}
#endif