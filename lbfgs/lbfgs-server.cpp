#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include "lbfgs_util.h"
#include "ThreadPool.h"

int THREAD_MAX = 10;
int TOTAL_NODE = 1;

ThreadPool pool(10);

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[]) {
    int sockfd, newsockfd, portno;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
       error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = 10086;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
             sizeof(serv_addr)) < 0) 
             error("ERROR on binding");
    listen(sockfd,5);
    clilen = sizeof(cli_addr);

    int i = 30;
    vector<future<double*>> results;
    vector<int> socks;
    int size;
    bool first = true;
    int SAMPLE_SIZE = 0;

    while(true) {
       printf("ready to get a conn\n");
       newsockfd = accept(sockfd, 
                   (struct sockaddr *) &cli_addr, 
                   &clilen);
       socks.push_back(newsockfd);
       recv(newsockfd, &first, sizeof(bool), 0);
       if(first) {
           int sampleSize;
           recv(newsockfd, &sampleSize, sizeof(int), 0);
           recv(newsockfd, &size, sizeof(int), 0);
           size = size + 1;
           SAMPLE_SIZE += sampleSize;
           printf("SAMPLESIZE  %d\n", SAMPLE_SIZE);
       } else {
           results.emplace_back(
               pool.enqueue( [newsockfd, &size] {
                   double* weight = (double*) malloc(size * sizeof(double));
                   int sum = 0;
                   while(sum != sizeof(double) * size) {
                       int n = recv(newsockfd, weight + sum/sizeof(double), sizeof(double) * size, 0);
                       sum += n;
                   }
                   printf("rtn norm:%f\n", norm(weight, size));
                   return weight;
               })
           );
       }
       if(socks.size() == TOTAL_NODE && results.size() == 0) {
           for(int sock : socks) {
               send(sock, &SAMPLE_SIZE, sizeof(int), 0);
               close(sock);
           }
           printf("SAMPLESIZE  %d\n", SAMPLE_SIZE);
           printf("size is %d\n", size);
           socks.clear();
           SAMPLE_SIZE = 0;
       }
       if(results.size() == TOTAL_NODE) {
           double* rtn = (double*) calloc(size, sizeof(double));
           for(auto && result : results) {
               double* res = result.get();
               for(int i = 0; i < size; ++i) {
                   rtn[i] += res[i];
               }
               free(res);
           }
           for(int sock : socks) {
               int n = send(sock, rtn, sizeof(double) * size, 0);
               printf("send n:%d\n", n);
               close(sock);
           }
           free(rtn);
           results.clear();
           socks.clear();
       }
    }
    close(sockfd);
    return 0; 
}
