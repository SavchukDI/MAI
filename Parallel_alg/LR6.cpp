#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>

typedef long long int lli;

lli mtime() {
	struct timeb t;
	ftime(&t);
	return t.time * 1000 + t.millitm;
}

#define _i(i, j) (((j) + 1) * (n + 2) + (i) + 1)

int main() {
	int i, j, n, it = 0;
	double lx, ly, hx, hy, bc_down, bc_up, bc_right, bc_left;
	double global_error, error, eps = 1e-4;
	double *data, *next, *temp;
	scanf("%d", &n);
	lx = 1.0;
	ly = 1.0;
	hx = lx / n;
	hy = ly / n;
	bc_down = 1.0;
	bc_up = 2.0;
	bc_right = 3.0;
	bc_left = 4.0;

	#ifdef _OPENMP
		fprintf(stderr, "USED OpenMP\n");
	#endif

	data = (double *)malloc(sizeof(double) * (n + 2) * (n + 2));
	next = (double *)malloc(sizeof(double) * (n + 2) * (n + 2));	
	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++)
			data[_i(i, j)] = 0.0;
		data[_i(-1, i)] = next[_i(-1, i)] = bc_left;
		data[_i(n, i)] = next[_i(n, i)] = bc_right;
		data[_i(i, -1)] = next[_i(i, -1)] = bc_down;
		data[_i(i, n)] = next[_i(i, n)] = bc_up;
	}
	lli start_time = mtime();
	do {
		it++;
		global_error = 0.0;
		#pragma omp parallel for private(i, j, error) shared(next, data, n) reduction(max: global_error)
		for(i = 0; i < n; i++)
			for(j = 0; j < n; j++) {
				next[_i(i, j)] = 0.5 * ((data[_i(i - 1, j)] + data[_i(i + 1, j)]) / pow(hx, 2.0) +
											(data[_i(i, j - 1)] + data[_i(i, j + 1)]) / pow(hy, 2.0)) / 
											(1.0 / pow(hx, 2.0) + 1.0 / pow(hy, 2.0));
				error = fabs(data[_i(i, j)] - next[_i(i, j)]);
				if (error > global_error)
					global_error = error;
			}
		temp = next;
		next = data;
		data = temp;
	} while(global_error > eps && it < 5000);
	fprintf(stderr, "it = %d, error = %e, time = %Ld\n", it, global_error, mtime() - start_time);

	for(j = 0; j < n; j++) {
		for(i = 0; i < n; i++)
			printf("%.2f ", data[_i(i, j)]);
		printf("\n");
	}

	free(data);
	free(next);
	return 0;
}