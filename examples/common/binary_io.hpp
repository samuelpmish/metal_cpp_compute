#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

template <typename T>
std::vector<T> read_binary(std::string filename) {
  std::vector<T> buffer;
  std::ifstream infile(filename, std::ios::binary);

  if (infile) {
    // std::cout << "file found: " << filename << std::endl;
    infile.seekg(0, std::ios::end);
    std::streampos filesize = infile.tellg();
    infile.seekg(0, std::ios::beg);

    buffer = std::vector<T>(filesize / sizeof(T));
    infile.read((char*)&buffer[0], filesize);
  } else {
    std::cout << "file not found: " << filename << std::endl;
  }

  infile.close();

  return buffer;
}

template <typename T>
void write_binary(const std::vector<T>& buffer, std::string filename) {
  std::ofstream outfile(filename, std::ios::binary);

  if (outfile) {
    outfile.write((char*)&buffer[0], sizeof(T) * buffer.size());
  } else {
    std::cout << "file not found: " << filename << std::endl;
  }

  outfile.close();
}