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
 * "Functional" version of pagerank, which is quite a bit more efficient, because
 * it does not construct the vertex-objects but directly processes the edges.
 *
 * This program can be run either in the semi-synchronous mode (faster, but
 * less clearly defined semantics), or synchronously. Synchronous version needs
 * double the amount of I/O because it needs to store both previous and 
 * current values. Use command line parameter mode with semisync or sync.
 */

#define RANDOMRESETPROB 0.15
#define GRAPHCHI_DISABLE_COMPRESSION

#define HTM
//#define RTM
//#define HLE

#include <string>
#include <fstream>
#include <cmath>

#include "graphchi_basic_includes.hpp"
#include "api/functional/functional_api.hpp"
#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

struct pagerank_kernel : public functional_kernel<float, float> {
    
    float init(graphchi_context &ginfo, vertex_info& vertex) {
        return 1.0;
    }
    
    float zero() {
        return 0.0;
    }
    
    float gather(graphchi_context &ginfo, vertex_info& vertex, vid_t nb_id, float nb_val) {
        return nb_val;
    }
    
    float plus(float acc, float toadd) {
        return acc + toadd;
    }
    
    float apply(graphchi_context &ginfo, vertex_info& vertex, float val, float sum) {
        assert(ginfo.nvertices > 0);
        return RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum;
    }
    
    float scatter(graphchi_context &ginfo, vertex_info& vertex, vid_t nb_id, float val) {
        assert(vertex.outdegree > 0);
        return val / vertex.outdegree;
    }
}; 

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("pagerank");
    
    std::string filename = get_option_string("file");
    int niters = get_option_int("niters", 4);
    bool onlytop = get_option_int("onlytop", 0);
    int ntop = get_option_int("top", 20);
    std::string mode = get_option_string("mode", "sync");
            
    if (onlytop == 0) {
        /* Run */
        if (mode == "semisync") {            
            logstream(LOG_INFO) << "Running pagerank in semi-synchronous mode." << std::endl;
            run_functional_unweighted_semisynchronous<pagerank_kernel>(filename, niters, m);
        } else if (mode == "sync") {
            logstream(LOG_INFO) << "Running pagerank in (bulk) synchronous mode." << std::endl;
            run_functional_unweighted_synchronous<pagerank_kernel>(filename, niters, m);
        } else {
            logstream(LOG_ERROR) << "Mode needs to be either 'semisync' or 'sync'." << std::endl;
            assert(false);
        }
        /* Output metrics */
        metrics_report(m);
    }
    
    /* Write Top 20 */
    std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
    std::cout << "Print top 20 vertices: " << std::endl;
    for(int i=0; i < (int) top.size(); i++) {
        std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
    }
    return 0;
}

