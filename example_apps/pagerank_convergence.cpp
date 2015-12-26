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

            ginfo.set_last_iteration(ginfo.iteration);
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

            /*
            volatile long x = 0;
            for(long i = 0; i < 5000; i++){
                x++;
                x--;
            }
*/
            //std::cout << "a" << std::endl;

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
};

/**
  * Faster version of pagerank which holds vertices in memory. Used only if the number
  * of vertices is small enough.
  */
struct PagerankProgramInmem : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    std::vector<EdgeDataType> pr;
    PagerankProgramInmem(int nvertices) :   pr(nvertices, RANDOMRESETPROB) {}

    void after_iteration(int iteration, graphchi_context &ginfo) {
        if(ginfo.iteration == 0 || ginfo.iteration == ginfo.last_iteration)
            return;

        for(int i = 0; i < ginfo.execthreads; i++)
            if(ginfo.deltas[i] > THRESHOLD) {
                ginfo.reset_deltas(ginfo.execthreads);
                return;
            }

        for(int i = 0; i < ginfo.execthreads; i++)
            logstream(LOG_INFO) << "delta" << i << " " << ginfo.deltas[i] << std::endl;

        ginfo.set_last_iteration(ginfo.iteration + 1);//hacky;
    }

    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
        if(ginfo.iteration == ginfo.last_iteration) {//write
            /* On last iteration, multiply pr by degree and store the result */
            v.set_data(v.outc > 0 ? pr[v.id()] * v.outc : pr[v.id()]);
        } else if (ginfo.iteration > 0) {
            float sum=0;
            for(int i=0; i < v.num_inedges(); i++) {
              sum += pr[v.inedge(i)->vertexid];
            }

            float pagerank = RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum;
            int thread_num = omp_get_thread_num();
            float pagerankcont = v.outc > 0 ? pagerank / v.outc : pagerank;
            float diff = std::abs(pagerankcont - pr[v.id()]);

            if(ginfo.deltas[thread_num] < diff){
                ginfo.deltas[thread_num] = diff;
            }
            pr[v.id()] = pagerankcont;
        } else if (ginfo.iteration == 0) {
            if (v.outc > 0) pr[v.id()] = 1.0f / v.outc;
        }
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
    graphchi_engine<float, float> engine(filename, nshards, scheduler, m); 
    engine.set_modifies_inedges(false); // Improves I/O performance.
    
    bool inmemmode = engine.num_vertices() * sizeof(EdgeDataType) < (size_t)engine.get_membudget_mb() * 1024L * 1024L;
    inmemmode = false;
    if (inmemmode) {
        logstream(LOG_INFO) << "Running Pagerank by holding vertices in-memory mode!" << std::endl;
        engine.set_modifies_outedges(false);
        engine.set_disable_outedges(true);
        engine.set_only_adjacency(true);
        PagerankProgramInmem program(engine.num_vertices());
        engine.run(program, niters);
    } else {
        PagerankProgram program;
        engine.run(program, niters);
    }

    /* Output top ranked vertices */
    std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
    std::cout << "Print top " << ntop << " vertices:" << std::endl;
    for(int i=0; i < (int)top.size(); i++) {
        std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
    }
    
    metrics_report(m);    
    return 0;
}

