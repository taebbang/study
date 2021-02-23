//https://taebbbang.tistory.com/31?category=798286
#include <iostream>
using namespace std;
int fibonacci(int n);
int arr[41] = { 0,1, };
int main(int argc, char** argv)
{
	int testcase,T,num;
	cin >> T;
	for (testcase = 0; testcase < T; testcase++)
	{
		cin >> num;
		if (num == 0) {
			cout << "1 0" << endl;
		}
		else {
			cout << fibonacci(num - 1) << " " << fibonacci(num) << endl;
		}
	}
	return 0;
}
int fibonacci(int n) {
	if (n == 0) {
		return arr[n];
	}
	else if (n == 1) {
		return arr[n];
	}
	else if (arr[n] > 1) {
		return arr[n];
	}
	else {
		return arr[n] = fibonacci(n-1) + fibonacci(n-2);
	}
}
