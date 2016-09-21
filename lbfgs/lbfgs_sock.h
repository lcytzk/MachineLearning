#ifndef __LBFGS_SOCK_H__
#define __LBFGS_SOCK_H__

#include <netdb.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>

string HOST_NAME = "shd-live-tmp-2";
void asyncLossAndGrad(double* lossAndGrad, int size) {
    int port = 10086;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct hostent* server = gethostbyname(HOST_NAME.c_str());
    struct sockaddr_in serv_addr;
    bzero((char*) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*) server->h_addr, (char*) &serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(port);

    if(connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connect error\n");
    }

    bool first = false;
    send(sockfd, &first, sizeof(bool), 0);
    send(sockfd, lossAndGrad, sizeof(double) * size, 0);

    int sum = 0;
    while(sum != sizeof(double) * size) {
        int n = recv(sockfd, lossAndGrad + sum/sizeof(double), sizeof(double) * size, 0);
        sum += n;
    }
}

void asyncSampleSizeAndWSize(int sampleSize, int wSize, int& allSample) {
    int port = 10086;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct hostent* server = gethostbyname(HOST_NAME.c_str());
    struct sockaddr_in serv_addr;
    bzero((char*) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*) server->h_addr, (char*) &serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(port);

    if(connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connect error\n");
    }

    bool first = true;
    send(sockfd, &first, sizeof(bool), 0);
    send(sockfd, &sampleSize, sizeof(int), 0);
    send(sockfd, &wSize, sizeof(int), 0);

    recv(sockfd, &allSample, sizeof(int), 0);
    printf("All example size is %d\n", allSample);
}

#endif // __LBFGS_SOCK_H__
