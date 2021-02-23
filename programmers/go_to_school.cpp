//https://taebbbang.tistory.com/29
#include <string>
#include <vector>
using namespace std;
int solution(int m, int n, vector<vector<int>> puddles) {
	int array[101][101] = { 0 };
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			array[i][j] = 1;
		}
	}
	for (int i = 0; i < puddles.size(); i++) {
		array[puddles[i][1]][puddles[i][0]] = 0;
		if (puddles[i][0] == 1) {
			for (int j = puddles[i][1]; j <= n; j++)
				check[j][1] = 0;
		}
		if (puddles[i][1] == 1) {
			for (int j = puddles[i][0]; j <= m; j++)
				check[1][j] = 0;
		}
	}

	for (int i = 2; i <= m; i++) {
		for (int j = 2; j <= n; j++) {
			if (array[j][i] != 0) {
				array[j][i] = (array[j - 1][i] + array[j][i - 1]) % 1000000007;
			}
		}
	}

	return array[n][m];
}
