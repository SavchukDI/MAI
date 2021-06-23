#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define _i(i, j) (((j) + 1) * (n + 2) + ((i) + 1))
#define _ib(i, j) ((j) * nb + (i))

int main(int argc, char *argv[]) {
    int it = 0, i, j, nb, n, ib, jb;
    double lx, ly, bc_down, bc_up, bc_left, bc_right, hx, hy, start_time, global_error, eps = 1e-5;
    double *data, *next, *buff;

    int id, numproc, proc_name_len;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);
    fprintf(stderr, "proc %d(%d) on %s\n", id, numproc, proc_name);
    fflush(stderr);

    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        scanf("%d %d", &nb, &n); // nbx nby nx ny
        lx = 1.0;
        ly = 1.0;
        bc_down = 1.0;
        bc_up = 2.0;
        bc_left = 3.0;
        bc_right = 4.0;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (nb * nb != numproc) {
        fprintf(stderr, "ERROR\n");
        MPI_Finalize();
        return 0;
    }

    hx = lx / (n * nb);
    hy = ly / (n * nb);
    
    ib = id % nb;
    jb = id / nb;

    data = (double *)malloc(sizeof(double) * (n + 2) * (n + 2));
    next = (double *)malloc(sizeof(double) * (n + 2) * (n + 2));
    buff = (double *)malloc(sizeof(double) * (n + 2));

    int buffer_size = 4 * ((n + 2) * sizeof(double) + MPI_BSEND_OVERHEAD);
    double *buffer = (double *)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            data[_i(i, j)] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0)
        start_time = MPI_Wtime();

    do {
        it++;
        if (ib + 1 < nb) {
            for(j = 0; j < n; j++)
                buff[j] = data[_i(n - 1, j)];
            MPI_Bsend(buff, n, MPI_DOUBLE, _ib(ib + 1, jb), id, MPI_COMM_WORLD);    
        }
        if (jb + 1 < nb) {
            for(j = 0; j < n; j++)
                buff[j] = data[_i(j, n - 1)];
            MPI_Bsend(buff, n, MPI_DOUBLE, _ib(ib, jb + 1), id, MPI_COMM_WORLD);    
        }
        if (ib > 0) {
            for(j = 0; j < n; j++)
                buff[j] = data[_i(0, j)];
            MPI_Bsend(buff, n, MPI_DOUBLE, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
        }
        if (jb > 0) {
            for(j = 0; j < n; j++)
                buff[j] = data[_i(j, 0)];
            MPI_Bsend(buff, n, MPI_DOUBLE, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
        }
        
        if (ib > 0) {
            MPI_Recv(buff, n, MPI_DOUBLE, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(j = 0; j < n; j++)
                data[_i(-1, j)] = buff[j];
        } else {
            for(j = 0; j < n; j++)
                data[_i(-1, j)] = bc_left;
        }
        
        if (jb > 0) {
            MPI_Recv(buff, n, MPI_DOUBLE, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(j = 0; j < n; j++)
                data[_i(j, -1)] = buff[j];
        } else {
            for(j = 0; j < n; j++)
                data[_i(j, -1)] = bc_down;
        }

        if (ib + 1 < nb) {
            MPI_Recv(buff, n, MPI_DOUBLE, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(j = 0; j < n; j++)
                data[_i(n, j)] = buff[j];
        } else {
            for(j = 0; j < n; j++)
                data[_i(n, j)] = bc_right;
        }

        if (jb + 1 < nb) {
            MPI_Recv(buff, n, MPI_DOUBLE, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(j = 0; j < n; j++)
                data[_i(j, n)] = buff[j];
        } else {
            for(j = 0; j < n; j++)
                data[_i(j, n)] = bc_up;
        }

        for(i = 0; i < n; i++)
            for(j = 0; j < n; j++)
                next[_i(i, j)] = 0.5 * ((data[_i(i - 1, j)] + data[_i(i + 1, j)]) / pow(hx, 2) + 
                                            (data[_i(i, j - 1)] + data[_i(i, j + 1)]) / pow(hy, 2)) / 
                                                (1.0 / pow(hx, 2) + 1.0 / pow(hy, 2));  
        double *temp = data;
        data = next;
        next = temp;

        double error, local_error = 0.0;
        for(i = 0; i < n; i++)
            for(j = 0; j < n; j++) {
                error = fabs(data[_i(i, j)] - next[_i(i, j)]);
                if (error > local_error)
                    local_error = error;
            }
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    } while (global_error > eps && it < 500);

    if (id == 0) {
        fprintf(stderr, "it = %d, error = %e, time = %e\n", it, global_error, MPI_Wtime() - start_time);
        fflush(stderr);
    }

    if (id != 0) {
        for(j = 0; j < n; j++) {
            for(i = 0; i < n; i++)
                buff[i] = data[_i(i, j)];
            MPI_Send(buff, n, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
        }
    } else {
        for(jb = 0; jb < nb; jb++)
            for(j = 0; j < n; j++)
                for(ib = 0; ib < nb; ib++) {
                    if (_ib(ib, jb) == 0) 
                        for(i = 0; i < n; i++)
                            buff[i] = data[_i(i, j)];
                    else 
                        MPI_Recv(buff, n, MPI_DOUBLE, _ib(ib, jb), _ib(ib, jb), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for(i = 0; i < n; i++)
                        printf("%.2f ", buff[i]);
                    if (ib + 1 == nb)// {
                        printf("\n");
                //      if (j + 1 == n)
                //          printf("\n");
                //  } else
                //      printf(" ");
                }
    }
    MPI_Buffer_detach(buffer, &buffer_size);
    MPI_Finalize();

    free(buffer);
    free(data);
    free(next);
    free(buff);
    return 0;
};