#define DYNAMICEDATA 1

#define GRAPHCHI_DISABLE_COMPRESSION

#include <string>
#include <fstream>
#include <cmath>
#include <set>
#include <algorithm>
#include <vector>

#include "graphchi_basic_includes.hpp"
#include "api/functional/functional_api.hpp"
#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

typedef int VT;//vertex type
typedef int EIT;//edge inner type
typedef chivector<EIT> ET;//edge type

std::vector<EIT> used_colors;

struct greedy_graph_coloring_kernel : public functional_kernel<VT, ET> {

    VT init(graphchi_context &ginfo, vertex_info& vertex) {
        //initial node
        if(vertex.vertexid == 0){
            used_colors.emplace_back(1);
            return 0;
        }
        return -1;//not colored yet
    }

    ET zero() {
        ET empty;
        return empty;
    }

    ET gather(graphchi_context &ginfo, vertex_info& vertex, vid_t nb_id, ET nb_val) {
        return nb_val;
    }

    ET plus(ET acc, ET toadd) {
        //merge chivector(intentionally, o(n^2))
        for(int i = 0; i < toadd.size(); i++){
            bool containing = false;
            for(int j = 0; j < acc.size(); j++){
                if(toadd[i] == acc[j]) {
                    containing = true;
                    break;
                }
            }
            if(!containing)
                acc.add(toadd[i]);
        }
        return acc;
    }

    VT apply(graphchi_context &ginfo, vertex_info& vertex, VT val, ET sum) {
        assert(ginfo.nvertices > 0);

        if(val > -1)//already colored
            return val;

        //get color
        for(int i = 0; i < used_colors.size(); i++){
            bool containing = false;
            for(int j = 0; j < sum.size(); j++){
                if(sum[j] == used_colors[i]){
                    containing = true;
                    break;
                }
            }

            if(!containing) {
                used_colors[i]++;
                return i;//use used color(not used by nb)
            }
        }

        used_colors.emplace_back(1);
        return used_colors.size() - 1;//new color
    }

    ET scatter(graphchi_context &ginfo, vertex_info& vertex, vid_t nb_id, VT val) {
        assert(vertex.outdegree > 0);
        ET cv;
        cv.add(val);
        return cv;
    }
};

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("greedy graph coloring");

    std::string filename = get_option_string("file");
    int niters = get_option_int("niters", 4);
    bool onlytop = get_option_int("onlytop", 0);
    int ntop = get_option_int("top", 20);
    std::string mode = get_option_string("mode", "sync");

    if (onlytop == 0) {
        /* Run */
        if (mode == "semisync") {
            logstream(LOG_INFO) << "Running pagerank in semi-synchronous mode." << std::endl;
            run_functional_unweighted_semisynchronous<greedy_graph_coloring_kernel>(filename, niters, m);
        } else if (mode == "sync") {
            logstream(LOG_INFO) << "Running pagerank in (bulk) synchronous mode." << std::endl;
            run_functional_unweighted_synchronous<greedy_graph_coloring_kernel>(filename, niters, m);
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