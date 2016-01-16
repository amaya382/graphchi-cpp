#define HTM
//#define RTM
//#define HLE

#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>

#include "graphchi_basic_includes.hpp"
#include "api/functional/functional_api.hpp"
#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

struct sssp_kernel : public functional_kernel<int, int> {

    int init(graphchi_context &ginfo, vertex_info& vertex) {
        return vertex.vertexid == 0 ? 0 : INT_MIN;
    }

    int zero() {
        return INT_MIN;
    }

    int gather(graphchi_context &ginfo, vertex_info& vertex, vid_t nb_id, int nb_val) {
        return nb_val;
    }

    int plus(int acc, int toadd) {
        if(acc < 0)
            return toadd;

        if(toadd < 0)
            return acc;

        return std::min(acc, toadd);
    }

    int apply(graphchi_context &ginfo, vertex_info& vertex, int val, int sum) {
        assert(ginfo.nvertices > 0);

        return sum < 0 ? val
                       : std::min(sum, val < 0 ? INT_MAX : val);
    }

    int scatter(graphchi_context &ginfo, vertex_info& vertex, vid_t nb_id, int val) {
        assert(vertex.outdegree > 0);
        return val + 1;
    }
};

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("sssp");

    std::string filename = get_option_string("file");
    int niters = get_option_int("niters", 100);
    bool onlytop = get_option_int("onlytop", 0);
    int ntop = get_option_int("top", 20);
    std::string mode = get_option_string("mode", "sync");

    if (onlytop == 0) {
        /* Run */
        if (mode == "semisync") {
            logstream(LOG_INFO) << "Running SSSP in semi-synchronous mode." << std::endl;
            run_functional_unweighted_semisynchronous<sssp_kernel>(filename, niters, m);
        } else if (mode == "sync") {
            logstream(LOG_INFO) << "Running SSSP in (bulk) synchronous mode." << std::endl;
            run_functional_unweighted_synchronous<sssp_kernel>(filename, niters, m);
        } else {
            logstream(LOG_ERROR) << "Mode needs to be either 'semisync' or 'sync'." << std::endl;
            assert(false);
        }
        /* Output metrics */
        metrics_report(m);
    }

    /* Write Top 20 */
    std::vector<vertex_value<int>> top = get_top_vertices<int>(filename, ntop);
    std::cout << "Print top 20 vertices: " << std::endl;
    for(int i=0; i < (int) top.size(); i++) {
        std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
    }
    return 0;
}