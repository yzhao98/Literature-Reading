//
//  main.cpp
//  mi
//
//  Created by 布衣匹夫 on 2019/6/29.
//  Copyright © 2019 布衣匹夫. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#define N 27
using namespace std;


struct Edge {
    int a;
    int b;
    double MI;
    bool operator < (const Edge &A) const
    {
        return MI > A.MI;
    }
}edge[N*(N-1)/2 + 2];

struct Tree {
    int a;
    int b;
    bool flag = true;
    bool operator < (const Edge &A) const
    {
        return a < A.a;
    }
}tree[N-2];

void init()
{
    ifstream f;
    f.open("/Users/Oasis/Desktop/exam/working_data/MI_c.txt");
//    printf("2");
//    if(f.good())
//        printf("is open");
    string str;
    vector<vector<double>> num;
    if(f.fail())
    {
        cout << "failed to read the file" << endl;
        return;
    }

    while(getline(f, str, ','))
    {
        istringstream input(str);
        vector<double> tmp;
        double a;
        while(input >> a)
            tmp.push_back(a);

        num.push_back(tmp);
    }

    int count = 1;
    for(int i = 2; i < N; i++)
    {
        for(int j = i + 1; j < N; j++)
        {
            edge[count].a = i;
            edge[count].b = j;
            edge[count].MI = num[i*N + j][0];
            printf("%d, %d, %d, %lf\n", count, i, j, edge[count].MI);
            count++;
        }
        cout << endl;
    }

    f.close();
    return;
}

int eHat[N]; // 0, 1, meaningful points
int eHat1[N];
//int findRoot(int x) // routine compress
//{
//    if (eHat[x] == -1)
//        return x;
//    else
//    {
//        int tmp = findRoot(eHat[x]);
//        eHat[x] = tmp;
//        return tmp;
//    }
//}

int findRoot(int x) // non-routine compress
{
    if (eHat[x] == -1)
        return x;
    else
        return findRoot(eHat[x]);
}

void display()
{
    for (int i = 2; i <= N - 1; i++)
    {
        printf("i:%d, parent:%d\n", i, eHat1[i]);
    }
    for (int i = 2; i <= N - 1; i++)
    {
        int tmp1 = i;
        while(eHat1[i] != -1)
        {
            printf("%d-", i);
            int tmp2 = i;
            i = eHat1[i];
            if(eHat1[i] == -1)
            {
                printf("%d\n", i);
            }
        }
        i = tmp1;
    }
}


int main() {
    init();
//    printf("1");
    
    sort(edge + 1, edge + (N-2)*(N-3)/2 + 1); // non-increasing
    for(int i = 1; i <= (N-2)*(N-3)/2 ; i++)
    {
        printf("%d, %d, %f\n", edge[i].a, edge[i].b, edge[i].MI);
    }
    
    // INIT
    for (int i = 2; i <= N - 1; i++)
    {
        eHat[i] = -1;
        eHat1[i] = -1;
    }
    int count = 1;
    double max_weight = 0;
    
    for (int i = 1; i <= (N-2)*(N-3)/2; i++)
    {
        printf("a: %d, a root: %d, b: %d, b root: %d\n",edge[i].a, findRoot(edge[i].a), edge[i].b, findRoot(edge[i].b));
        int a = findRoot(edge[i].a);
        int b = findRoot(edge[i].b);
        if (a != b) // if not cycle, add it.
        {
            eHat[a] = b;
            tree[count].a = edge[i].a;
            tree[count].b = edge[i].b;
            printf("...%d,%d\n", eHat[a], eHat1[a]);
            max_weight += edge[i].MI;
            count++;
        }
        else
        if(count == N - 2)
            break;
    }
    
    printf("Max Weight is %f\n", max_weight);
//    display();
    for (int i = 1; i < N-2; i++)
    {
        printf("%d-%d\n", tree[i].a, tree[i].b);

    }
    return 0;
}
