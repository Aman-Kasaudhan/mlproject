#include <bits/stdc++.h>
using namespace std;
int main(){
    int t;
    cin>>t;
    while(t--){
        int n,m,h;
        cin>>n>>m>>h;
        vector<int>arr(n);
        // vector<int>og(n);
        // int og[n];
        for(int i=0;i<n;i++){
            cin>>arr[i];
        }
        // og=arr;
       
        int b1,c1;
        for(int i=0;i<m;i++){
            cin>>b1>>c1;
            arr[b1-1]+=c1;
            if(arr[b1-1]>h){
                arr[b1-1]-=c1;
            }

        }

        for(int i=0;i<n;i++)
        cout<<arr[i]<<" ";
        cout<<endl;
    }
}