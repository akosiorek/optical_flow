#include <iostream>
#include <vector>
//#include <memory>

class MyClass {};

int main(int argc, char** argv) {

	std::vector<int> vec;
	for(int i = 0; i < 15; ++i) {
		vec.push_back(i * 3^i / 10);
	}

	std::cout << vec[0] << std::endl;
	
	std::vector<std::shared_ptr<MyClass>> vec2;
	return 0;
}
