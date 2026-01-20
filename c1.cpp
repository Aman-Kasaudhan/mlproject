#include<bits/stdc++.h>
using namespace std;

int main(){
    int t;
    cin>>t;
    while(t--){
        int n;
        cin>>n;

        vector<int>arr(n);

        for (int i = 0; i < n; i++)
        {
            cin>>arr[i];
        }

        vector<int>mp;
        vector<int>mp2;
        unordered_map<int,int>mp1;
        unordered_map<int,int>mp3;

        int j=0;
        for(int i=0;i<n-1;i++){
            int a=1;
            while(a){
                if(arr[i]!=j && mp1.find(j)==mp1.end()){
                    mp.push_back(j);
                    a=0;
                    mp1[arr[i]]++;
                    
                }
                else j++;

            }
        }


        int j1=0;
        for(int i=n-1;i>0;i--){
            int a=1;
            while(a){
                if(arr[i]!=j1 && mp3.find(j1)==mp3.end()){
                    mp2.push_back(j1);
                    a=0;
                    mp3[arr[i]]++;
                    
                }
                else j1++;

            }
        }
         int w=1;
        for(int i=0;i<n-1;i++){
            if(mp[i]!=mp2[n-2-i] && w){
                cout<<"yes"<<endl;
                   w=0;
            }
        }

        if(w)
        cout<<"no"<<endl;
        
        
    }
}