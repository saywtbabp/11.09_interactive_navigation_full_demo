// Small C ABI: no Python/NumPy ABI dependency, no scene truth or dynamics.
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>
#include <functional>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
API int raster(int nx,int ny,double ox,double oy,double step,int count,const double* boxes,uint8_t* blocked){
 if(nx<1||ny<1||step<=0||count<0||!blocked)return -1;
 for(int x=0;x<nx;++x)for(int y=0;y<ny;++y){
  const double px=ox+x*step,py=oy+y*step;bool hit=false;
  for(int k=0;k<count&&!hit;++k){const double* b=boxes+8*k;double dx=px-b[0],dy=py-b[1];
   hit=std::abs(dx*b[2]+dy*b[4])<=b[6]+1e-9&&std::abs(dx*b[3]+dy*b[5])<=b[7]+1e-9;}
  blocked[x*ny+y]=hit;
 }return 0;
}
API int distances(int nx,int ny,double step,const uint8_t* blocked,int goal,double* dist){
 if(nx<1||ny<1||step<=0||!blocked||!dist||goal<0||goal>=nx*ny||blocked[goal])return -1;
 try{
  const double inf=std::numeric_limits<double>::infinity();for(int i=0;i<nx*ny;++i)dist[i]=inf;
  using Node=std::pair<double,int>;std::priority_queue<Node,std::vector<Node>,std::greater<Node>> queue;
  dist[goal]=0.;queue.push({0.,goal});
  const int dxs[8]={1,-1,0,0,1,1,-1,-1},dys[8]={0,0,1,-1,1,-1,1,-1};
  while(!queue.empty()){
   auto node=queue.top();queue.pop();if(node.first>dist[node.second])continue;
   int x=node.second/ny,y=node.second%ny;
   for(int k=0;k<8;++k){int dx=dxs[k],dy=dys[k],xx=x+dx,yy=y+dy;if(xx<0||xx>=nx||yy<0||yy>=ny)continue;
    int j=xx*ny+yy;if(blocked[j]||blocked[(x+dx)*ny+y]||blocked[x*ny+y+dy])continue;
    double value=node.first+step*(dx&&dy?std::sqrt(2.):1.);
    if(value<dist[j]){dist[j]=value;queue.push({value,j});}
   }
  }return 0;
 }catch(...){return -2;}
}
API int visible(int segments,int count,const double* boxes,const double* starts,const double* ends,uint8_t* result){
 if(segments<0||count<0||!result)return -1;
 for(int i=0;i<segments;++i){bool clear=true;
  for(int k=0;k<count&&clear;++k){const double* b=boxes+8*k;
   double ax=starts[2*i]-b[0],ay=starts[2*i+1]-b[1],bx=ends[2*i]-b[0],by=ends[2*i+1]-b[1];
   double a[2]={ax*b[2]+ay*b[4],ax*b[3]+ay*b[5]},z[2]={bx*b[2]+by*b[4],bx*b[3]+by*b[5]};
   double lo=0.,hi=1.;bool intersects=true;
   for(int d=0;d<2;++d){double v=z[d]-a[d],h=b[6+d]+1e-9;
    if(std::abs(v)<1e-14){if(std::abs(a[d])>h)intersects=false;}
    else{double t0=(-h-a[d])/v,t1=(h-a[d])/v;if(t0>t1)std::swap(t0,t1);lo=std::max(lo,t0);hi=std::min(hi,t1);if(lo>hi)intersects=false;}
   }if(intersects)clear=false;
  }result[i]=clear;
 }return 0;
}
