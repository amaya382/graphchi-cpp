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
 * Simple pagerank implementation. Uses the basic vertex-based API for
 * demonstration purposes. A faster implementation uses the functional API,
 * "pagerank_functional".
 */

#include <string>
#include <fstream>
#include <cmath>

#define GRAPHCHI_DISABLE_COMPRESSION


#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;
 
#define THRESHOLD 1e-2
#define RANDOMRESETPROB 0.15


typedef float VertexDataType;
typedef float EdgeDataType;

struct PagerankProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    /**
      * Called before an iteration starts. Not implemented.
      */
    void before_iteration(int iteration, graphchi_context &info) {
    }
    
    /**
      * Called after an iteration has finished. Not implemented.
      */
    void after_iteration(int iteration, graphchi_context &ginfo) {
        if(ginfo.iteration == 0)
            return;

        for(int i = 0; i < ginfo.execthreads; i++)
            if(ginfo.deltas[i] > THRESHOLD) {
                ginfo.reset_deltas(ginfo.execthreads);
                return;
            }

        for(int i = 0; i < ginfo.execthreads; i++)
            logstream(LOG_INFO) << "delta" << i << " " << ginfo.deltas[i] << std::endl;

            //ginfo.set_last_iteration(ginfo.iteration);
    }
    
    /**
      * Called before an execution interval is started. Not implemented.
      */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {        
    }

    virtual bool repeat_updates(graphchi_context &gcontext) {
        return false;
    }
    
    /**
      * Pagerank update function.
      */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
        float sum=0;
        if (ginfo.iteration == 0) {
            /* On first iteration, initialize vertex and out-edges. 
               The initialization is important,
               because on every run, GraphChi will modify the data in the edges on disk. 
             */
            for(int i=0; i < v.num_outedges(); i++) {
                graphchi_edge<float> * edge = v.outedge(i);
                edge->set_data(1.0 / v.num_outedges());
            }
            v.set_data(RANDOMRESETPROB);
        } else {
            /* Compute the sum of neighbors' weighted pageranks by
               reading from the in-edges. */
            for(int i=0; i < v.num_inedges(); i++) {
                float val = v.inedge(i)->get_data();
                sum += val;                    
            }
            
            /* Compute my pagerank */
            float pagerank = RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum;
            
            /* Write my pagerank divided by the number of out-edges to
               each of my out-edges. */
            if (v.num_outedges() > 0) {
                float pagerankcont = pagerank / v.num_outedges();
                for(int i=0; i < v.num_outedges(); i++) {
                    graphchi_edge<float> * edge = v.outedge(i);
                    edge->set_data(pagerankcont);
                }
            }

#define LOAD
#ifdef LOAD
            volatile long x = 0;
            for(long i = 0; i < 500; i++){
                x++;
                x--;
            }
#endif

            float diff = std::abs(pagerank - v.get_data());
            int thread_num = omp_get_thread_num();
            if(ginfo.deltas[thread_num] < diff){
                ginfo.deltas[thread_num] = diff;
            }

            /* Keep track of the progression of the computation.
               GraphChi engine writes a file filename.deltalog. */
            //ginfo.log_change(diff);
            
            /* Set my new pagerank as the vertex value */
            v.set_data(pagerank); 
        }
    }

    VertexDataType init(graphchi_context &info, graphchi_vertex<VertexDataType, EdgeDataType> &myvertex) {
        for(int i=0; i < myvertex.num_outedges(); i++) {
            auto edge = myvertex.outedge(i);
            edge->set_data(1.0 / myvertex.num_outedges());
        }
        myvertex.set_data(RANDOMRESETPROB);
        return 1.0;
    }

    VertexDataType zero() {
        return 0.0;
    }

    VertexDataType gather(graphchi_context &info, VertexDataType nbval) {
        return nbval;
    }

    VertexDataType sum(VertexDataType curval, VertexDataType toadd) {
        return curval + toadd;
    }

    VertexDataType apply(graphchi_context &ginfo, graphchi_vertex<VertexDataType, EdgeDataType> &myvertex, VertexDataType nbvalsum) {
        auto pagerank = RANDOMRESETPROB + (1 - RANDOMRESETPROB) * nbvalsum;
        if(myvertex.num_outedges() > 0){
            auto pagerank_cont = pagerank / myvertex.num_outedges();
            for(int i=0; i < myvertex.num_outedges(); i++) {
                auto edge = myvertex.outedge(i);
                edge->set_data(pagerank_cont);
            }
        }
        myvertex.set_data(pagerank);
        return pagerank;
    }

    // "Scatter
    VertexDataType scatter(graphchi_context &info, graphchi_vertex<VertexDataType, EdgeDataType> &myvertex, vid_t nbid, VertexDataType myval) {
        return myval / myvertex.num_outedges();
    }
};

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("pagerank");
    global_logger().set_log_level(LOG_DEBUG);

    /* Parameters */
    std::string filename    = get_option_string("file"); // Base filename
    int niters              = get_option_int("niters", 4);
    bool scheduler          = false;                    // Non-dynamic version of pagerank.
    int ntop                = get_option_int("top", 20);
    
    /* Process input file - if not already preprocessed */
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));

    /* Run */
    graphchi_engine_new<float, float> engine(filename, nshards, scheduler, m);
    engine.set_modifies_inedges(false); // Improves I/O performance.
    

        PagerankProgram program;
        engine.run(program, niters);


    /* Output top ranked vertices */
    std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
    std::cout << "Print top " << ntop << " vertices:" << std::endl;
    for(int i=0; i < (int)top.size(); i++) {
        std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
    }
    
    metrics_report(m);    
    return 0;
}

