//https://taebbbang.tistory.com/30?category=798286
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int row, col;
int Answer;
int arr[501][501];
int dp[501][501];
int a[4] = { 1, 0, -1, 0 };
int b[4] = { 0, 1, 0, -1 };
int dfs( int x, int y);
int main(int argc, char** argv)
{
	cin >> row;
	cin >> col;
	for (int i = 0; i <row; i++)
	{
		for(int j = 0; j <col ; j ++){
			cin >> arr[i][j];
			dp[i][j] = -1;
		}
	}
	cout << dfs(row - 1, col - 1);
	return 0;
}
int dfs(int x, int y) {
	if (dp[x][y] != -1) return dp[x][y]; 
	if (x < 0 || x >= row || y < 0 || y >= col) return 0; 
	if (x == 0 && y == 0) return 1; 

	dp[x][y] = 0;
	for (int i = 0; i < 4; i++)
	{
		int nextX = x + a[i];
		int nextY = y + b[i]; 

		if (arr[nextX][nextY] > arr[x][y])
			dp[x][y] += dfs(nextX, nextY);
	}

	return dp[x][y];
}
