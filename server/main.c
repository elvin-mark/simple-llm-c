#include "llm/models/gpt2.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PORT 8080
#define BUFFER_SIZE 4096
#define CTX_MAX_SIZE 1024
#define INDEX_HTML_FILE "server/index.html"

GPT2 *gpt2;

void process_tokens(char *input, char *output) {
  int tokens[CTX_MAX_SIZE], count = 0;
  int new_tokens_count;
  char *token = strtok(input, ",");

  new_tokens_count = atoi(token);
  token = strtok(NULL, ",");

  while (token != NULL && count < CTX_MAX_SIZE) {
    tokens[count] = atoi(token);
    token = strtok(NULL, ",");
    count++;
  }

  gpt2_generate(&count, tokens, *gpt2, 12, new_tokens_count);

  int offset = 0;
  for (int i = 0; i < count; i++)
    offset +=
        sprintf(output + offset, "%d%s", tokens[i], (i < count - 1) ? "," : "");
}

void serve_index(int client_socket) {
  int fd = open(INDEX_HTML_FILE, O_RDONLY);
  if (fd == -1) {
    const char *not_found_response = "HTTP/1.1 404 Not Found\r\n"
                                     "Content-Type: text/plain\r\n"
                                     "Content-Length: 13\r\n"
                                     "\r\n"
                                     "404 Not Found";
    send(client_socket, not_found_response, strlen(not_found_response), 0);
    return;
  }

  struct stat file_stat;
  fstat(fd, &file_stat);
  size_t file_size = file_stat.st_size;

  char headers[BUFFER_SIZE];
  snprintf(headers, sizeof(headers),
           "HTTP/1.1 200 OK\r\n"
           "Content-Type: text/html\r\n"
           "Content-Length: %ld\r\n"
           "\r\n",
           file_size);
  send(client_socket, headers, strlen(headers), 0);

  char file_buffer[BUFFER_SIZE];
  ssize_t bytes_read;
  while ((bytes_read = read(fd, file_buffer, sizeof(file_buffer))) > 0) {
    send(client_socket, file_buffer, strlen(file_buffer), 0);
  }
  close(fd);
}

void handle_client(int client_socket) {
  char buffer[BUFFER_SIZE], response[BUFFER_SIZE];
  int recv_len = recv(client_socket, buffer, BUFFER_SIZE - 1, 0);

  if (recv_len <= 0) {
    close(client_socket);
    return;
  }

  buffer[recv_len] = '\0';

  if (strncmp(buffer, "GET / ", 6) == 0) {
    serve_index(client_socket);
  } else if (strncmp(buffer, "POST / ", 6) == 0) {

    // Look for the start of the body in the HTTP request
    char *body = strstr(buffer, "\r\n\r\n");
    if (!body) {
      close(client_socket);
      return;
    }
    body += 4; // Move past the "\r\n\r\n" to get to the actual data

    // Process tokens
    char output[BUFFER_SIZE] = {0};
    process_tokens(body, output);

    // Prepare HTTP response
    snprintf(response, sizeof(response),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: text/plain\r\n"
             "Content-Length: %ld\r\n"
             "\r\n"
             "%s",
             strlen(output), output);

    send(client_socket, response, strlen(response), 0);
  } else {
    const char *not_found_response = "HTTP/1.1 404 Not Found \r\n"
                                     "Content-Type: text/plain\r\n"
                                     "Content-Length: 13\r\n"
                                     "\r\n"
                                     "404 Not Found";
    send(client_socket, not_found_response, strlen(not_found_response), 0);
  }
  close(client_socket);
}

int main() {
  int server_fd, client_socket;
  struct sockaddr_in server_addr, client_addr;
  socklen_t addr_len = sizeof(client_addr);

  // Create socket
  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == -1) {
    perror("Socket creation failed");
    exit(EXIT_FAILURE);
  }

  // Bind socket
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(PORT);

  if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) ==
      -1) {
    perror("Bind failed");
    close(server_fd);
    exit(EXIT_FAILURE);
  }

  // Listen for connections
  if (listen(server_fd, 5) == -1) {
    perror("Listen failed");
    close(server_fd);
    exit(EXIT_FAILURE);
  }

  GPT2Config config = {768, 12, 50257, 1024, 3072, 12};
  gpt2 = load_model(config, "model.bin");

  printf("Server listening on port %d...\n", PORT);

  while (1) {
    // Accept client connection
    client_socket =
        accept(server_fd, (struct sockaddr *)&client_addr, &addr_len);
    if (client_socket == -1) {
      perror("Accept failed");
      continue;
    }

    handle_client(client_socket);
  }

  close(server_fd);
  free_model(gpt2);
  return 0;
}
