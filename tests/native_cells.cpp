#include "meshing/cell.h"
#include <iostream>
#include <stdexcept>
using fuse::detail::Cell;
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
int main() {
    try {
        // Every sign case, unequal magnitudes, and both exact and nonzero face
        // determinants. Each crossing belongs to exactly one closed contour.
        for(int mask=0;mask<256;++mask) for(int variant=0;variant<4;++variant) {
            Cell c;
            for(int i=0;i<8;++i) c.values[i]=((mask>>i)&1 ? -1:1)*(variant ? 0.2+(i*7+variant*11)%13:1.0);
            c.build();unsigned seen=0;int loop[12],crossings=0,visited=0;
            for(int e=0;e<12;++e) {
                int a,b;Cell::endpoints(e,a,b);bool active=(c.values[a]<0)!=(c.values[b]<0);
                require(c.degree[e]==(active?2:0),"invalid contour degree");
                crossings+=active;
            }
            for(int e=0;e<12;++e) if(c.degree[e] && !(seen&(1u<<e))) {
                int n=c.loop(e,loop,seen);require(n>=3 && n<=12,"invalid loop length");visited+=n;
                for(int k=0;k<n;++k) {
                    int a=loop[k],b=loop[(k+1)%n];
                    require(c.adjacent[a][0]==b || c.adjacent[a][1]==b,"loop does not close");
                }
            }
            require(visited==crossings,"crossing lost or duplicated");
        }
        std::cout<<"1024 cube contour configurations passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
