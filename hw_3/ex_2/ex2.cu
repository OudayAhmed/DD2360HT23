__global__ void mover_kernel(
    struct parameters param, struct grid grd, int nop, FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w,
    FPfield* Ex, FPfield* Ey, FPfield* Ez, FPfield* Bxn, FPfield* Byn, FPfield* Bzn,
    FPfield* XN, FPfield* YN, FPfield* ZN, int n_sub_cycles, FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    int NiterMover
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nop) {
        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
        xptilde = x[i];
        yptilde = y[i];
        zptilde = z[i];

        FPpart omdtsq, denom, ut, vt, wt, udotb;

        // local (to the particle) electric and magnetic field
        FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        // interpolation densities
        int ix, iy, iz;
        FPfield weight[2][2][2];
        FPfield xi[2], eta[2], zeta[2];


        // start subcycling
        for (int i_sub = 0; i_sub < n_sub_cycles; i_sub++) {

            // calculate the average velocity iteratively
            for (int innter = 0; innter < NiterMover; innter++) {
                // interpolation G-->P
                ix = 2 + int((x[i] - grd.xStart) * grd.invdx);
                iy = 2 + int((y[i] - grd.yStart) * grd.invdy);
                iz = 2 + int((z[i] - grd.zStart) * grd.invdz);

                // calculate weights
                xi[0] = x[i] - XN[get_idx(ix - 1, iy, iz, grd.nyn, grd.nzn)];
                eta[0] = y[i] - YN[get_idx(ix, iy - 1, iz, grd.nyn, grd.nzn)];
                zeta[0] = z[i] - ZN[get_idx(ix, iy, iz - 1, grd.nyn, grd.nzn)];
                xi[1] = XN[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - x[i];
                eta[1] = YN[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - y[i];
                zeta[1] = ZN[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - z[i];

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;

                // set to zero local electric and magnetic field
                Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++) {
                            Exl += weight[ii][jj][kk] * Ex[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                            Eyl += weight[ii][jj][kk] * Ey[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                            Ezl += weight[ii][jj][kk] * Ez[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                            Bxl += weight[ii][jj][kk] * Bxn[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                            Byl += weight[ii][jj][kk] * Byn[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                            Bzl += weight[ii][jj][kk] * Bzn[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                        }
                // end interpolation
                omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
                denom = 1.0 / (1.0 + omdtsq);

                // solve the position equation
                ut = u[i] + qomdt2 * Exl;
                vt = v[i] + qomdt2 * Eyl;
                wt = w[i] + qomdt2 * Ezl;
                udotb = ut * Bxl + vt * Byl + wt * Bzl;

                // solve the velocity equation
                uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
                vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
                wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

                // update position
                x[i] = xptilde + uptilde * dto2;
                y[i] = yptilde + vptilde * dto2;
                z[i] = zptilde + wptilde * dto2;

            } // end of iteration

              // update the final position and velocity
            u[i] = 2.0 * uptilde - u[i];
            v[i] = 2.0 * vptilde - v[i];
            w[i] = 2.0 * wptilde - w[i];
            x[i] = xptilde + uptilde * dt_sub_cycling;
            y[i] = yptilde + vptilde * dt_sub_cycling;
            z[i] = zptilde + wptilde * dt_sub_cycling;

            //////////
            //////////
            ////////// BC

            // X-DIRECTION: BC particles
            if (x[i] > grd.Lx) {
                if (param.PERIODICX == true) { // PERIODIC
                    x[i] = x[i] - grd.Lx;
                }
                else { // REFLECTING BC
                    u[i] = -u[i];
                    x[i] = 2 * grd.Lx - x[i];
                }
            }

            if (x[i] < 0) {
                if (param.PERIODICX == true) { // PERIODIC
                    x[i] = x[i] + grd.Lx;
                }
                else { // REFLECTING BC
                    u[i] = -u[i];
                    x[i] = -x[i];
                }
            }

            // Y-DIRECTION: BC particles
            if (y[i] > grd.Ly) {
                if (param.PERIODICY == true) { // PERIODIC
                    y[i] = y[i] - grd.Ly;
                }
                else { // REFLECTING BC
                    v[i] = -v[i];
                    y[i] = 2 * grd.Ly - y[i];
                }
            }

            if (y[i] < 0) {
                if (param.PERIODICY == true) { // PERIODIC
                    y[i] = y[i] + grd.Ly;
                }
                else { // REFLECTING BC
                    v[i] = -v[i];
                    y[i] = -y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (z[i] > grd.Lz) {
                if (param.PERIODICZ == true) { // PERIODIC
                    z[i] = z[i] - grd.Lz;
                }
                else { // REFLECTING BC
                    w[i] = -w[i];
                    z[i] = 2 * grd.Lz - z[i];
                }
            }

            if (z[i] < 0) {
                if (param.PERIODICZ == true) { // PERIODIC
                    z[i] = z[i] + grd.Lz;
                }
                else { // REFLECTING BC
                    w[i] = -w[i];
                    z[i] = -z[i];
                }
            }
        }
    }
}



int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param) {
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    FPpart* d_x, * d_y, * d_z, * d_u, * d_v, * d_w;
    FPfield* d_Ex, * d_Ey, * d_Ez, * d_Bxn, * d_Byn, * d_Bzn, * d_XN_flat, * d_YN_flat, * d_ZN_flat;
    cudaMalloc(&d_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_w, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_Ex, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_Ey, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_Ez, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_Bxn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_Byn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_Bzn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(d_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ex, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ey, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ez, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bxn, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Byn, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bzn, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;

    int num_part = part->nop;
    int Db = 512;
    int Dg = (num_part + Db - 1) / Db;

    mover_kernel << <Dg, Db >> > (
        *param, *grd, num_part, d_x, d_y, d_z, d_u, d_v, d_w,
        d_Ex, d_Ey, d_Ez, d_Bxn, d_Byn, d_Bzn, d_XN_flat, d_YN_flat, d_ZN_flat,
        part->n_sub_cycles, dt_sub_cycling, dto2, qomdt2, part->NiterMover
        );

    cudaDeviceSynchronize();

    cudaMemcpy(part->x, d_x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, d_y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, d_z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, d_u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, d_v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, d_w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ex_flat, d_Ex, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, d_Ey, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, d_Ez, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, d_Bxn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, d_Byn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, d_Bzn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->XN_flat, d_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->YN_flat, d_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->ZN_flat, d_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_Ex);
    cudaFree(d_Ey);
    cudaFree(d_Ez);
    cudaFree(d_Bxn);
    cudaFree(d_Byn);
    cudaFree(d_Bzn);
    cudaFree(d_XN_flat);
    cudaFree(d_YN_flat);
    cudaFree(d_ZN_flat);

    return 0;
}