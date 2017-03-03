#include "stdafx.h"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <tuple>
#include <algorithm>
#include <pmmintrin.h>
#include <iomanip>

using namespace std;
namespace ffmtypes {
	typedef vector<vector<vector<float>>> matrix3d;
	typedef vector<vector<float>>         matrix2d;
	typedef vector<vector<int>>            intmatrix;
	typedef vector<int>                    intvector;
}

using namespace ffmtypes;
class FFM {
public:
	float eta;
	int* X, *Xval;
	intvector y, yval;
	int tr_rowcnt, val_rowcnt;
	float* lw, *lgradacc;

	FFM(int k = 4, float L2 = 0.00001, float eta = 0.2) :
		k{ k },
		eta{ eta },
		L2{ L2 }
	{}

	void initialize() {

		default_random_engine generator;
		normal_distribution<float> distribution(0.0, 0.5);

		lw = malloc_aligned_float((long)nfeat*nfield*k);
		lgradacc = malloc_aligned_float((long)nfeat*nfield*k);

		float* ww = lw;
		float* lgr = lgradacc;

		for (auto f = 0; f < nfeat; f++) {

			for (auto c = 0; c < nfield; c++) {

				for (auto w = 0; w < k; w++, ww++, lgr++) {
					*ww = distribution(generator);
					*lgr = 1;
				}
			}
		}
		cout << "Weights were set\n";
	}

#pragma warning(disable : 4996)
	tuple<int, int, int> analyze_data(char* pth, int maxrowcount) {
		int num_rows, maxfeat, maxfield;
		
		FILE *f = fopen(pth, "r");
		vector<char> str(1000);

		char* x;

		for (num_rows = 0; fgets(str.data(), 1000, f) != nullptr; num_rows++) {
			strtok(str.data(), ",");
			for (int j = 1;; j++) {
				x = strtok(nullptr, ",");
				if (x == nullptr || *x == '\n') {
					break;
				}
				else {
					maxfeat = max(maxfeat, atoi(x));
					maxfield = max(maxfield, j);
				};
			}

			if (num_rows >= maxrowcount) {
				break;
			}
		}

		fclose(f);

		maxfeat += 1;
		return { num_rows, maxfeat, maxfield };
	}

	void read_data(char* pth, int* data, intvector& label) {
		FILE *f = fopen(pth, "r");
		vector<char> str(1000);

		char* x;

		for (int i = 0; fgets(str.data(), 1000, f) != nullptr & i < tr_rowcnt; i++) {
			x = strtok(str.data(), ",");
			label[i] = atoi(x);
			for (int j = 0;; j++) {
				x = strtok(nullptr, ",");
				if (x == nullptr || *x == '\n') {
					break;
				}
				else {
					*data = atoi(x);
					data++;
				};
			}
		}

		fclose(f);
	}

	void load_data(char* trpth, char* valpth = nullptr, int num_rows = 1000) {
		tuple<int, int, int> data_stats = analyze_data(trpth, num_rows);
		tr_rowcnt = get<0>(data_stats);
		nfeat     = get<1>(data_stats);
		nfield    = get<2>(data_stats);

		y = intvector(tr_rowcnt);
		X = new int[tr_rowcnt*nfield];

		read_data(trpth, X, y);
		
		cout << "Train data loaded.\nNumber of rows: " << tr_rowcnt << "\nNumber of features: " << nfeat << endl;

		if (valpth) {
			tuple<int, int, int> val_stats = analyze_data(valpth, num_rows);
			val_rowcnt     = get<0>(val_stats);
			int nfeat_val  = get<1>(val_stats);
			int nfield_val = get<2>(val_stats);

			yval = intvector(val_rowcnt);
			Xval = new int[val_rowcnt*nfield_val];

			read_data(valpth, Xval, yval);

			cout << "Validate data loaded.\nNumber of rows: " << val_rowcnt << "\nNumber of features: " << nfeat_val << endl;
		}

		initialize();
	}

	float getz(int idx, int* data) {
		float z = 0;
		__m128 XMMw1, XMMw2, XMMinter;
		XMMinter = _mm_setzero_ps();
		int* train = data;
		int* pf1 = train + idx*nfield;

		for (int i = 0; i < nfield - 1; i++, pf1++) {
			int f1 = *pf1;
			if(f1 < nfeat)
				for (int j = i + 1, *pf2=pf1+1; j < nfield; j++, pf2++) {
					int f2 = *pf2;

					if (f2 < nfeat) {
						float* w1 = lw + f1*nfield*k + j*k;
						float* w2 = lw + f2*nfield*k + i*k;

						for (int l = 0; l < k; l += 4) {
							XMMw1 = _mm_load_ps(w1 + l);
							XMMw2 = _mm_load_ps(w2 + l);
							XMMinter = _mm_add_ps(XMMinter, _mm_mul_ps(XMMw1, XMMw2));
						}
					}
				}
		}
		XMMinter = _mm_hadd_ps(XMMinter, XMMinter);
		XMMinter = _mm_hadd_ps(XMMinter, XMMinter);
		_mm_store_ss(&z, XMMinter);
		return z;
	}

	float getdelta(int idx) {
		return (float)-y[idx] / (1 + exp(y[idx] * getz(idx, X)));
	}

	void setgrad(int idx) {
		float delta = getdelta(idx);
		__m128 XMMdelta = _mm_set1_ps(delta);
		__m128 lg1, lg2, w1, w2, gacc1, gacc2;
		__m128 XMML2 = _mm_set1_ps(L2);
		__m128 XMMeta = _mm_set1_ps(eta);
		int* train = X;

		for (int i = 0; i < nfield - 1; i++) {
			int f1 = *(train + idx*nfield + i);

			for (int j = i + 1; j < nfield; j++) {
				int f2 = *(train + idx*nfield + j);

				float* W1 = lw + f1*nfield*k + j*k;
				float* W2 = lw + f2*nfield*k + i*k;
				float* acc1 = lgradacc + f1*nfield*k + j*k;
				float* acc2 = lgradacc + f2*nfield*k + i*k;

				for (int l = 0; l < k; l+=4) {
					gacc1 = _mm_load_ps(acc1+l);
					gacc2 = _mm_load_ps(acc2+l);

					w1 = _mm_load_ps(W1+l);
					w2 = _mm_load_ps(W2+l);

					lg1 = _mm_add_ps(_mm_mul_ps(XMML2, w1), _mm_mul_ps(XMMdelta, w2));
					lg2 = _mm_add_ps(_mm_mul_ps(XMML2, w2), _mm_mul_ps(XMMdelta, w1));

					gacc1 = _mm_add_ps(gacc1, _mm_mul_ps(lg1, lg1));
					gacc2 = _mm_add_ps(gacc2, _mm_mul_ps(lg2, lg2));

					w1 = _mm_sub_ps(w1, _mm_mul_ps(XMMeta,
						_mm_mul_ps(lg1,
							_mm_rsqrt_ps(gacc1))));
					w2 = _mm_sub_ps(w2, _mm_mul_ps(XMMeta,
						_mm_mul_ps(lg2,
							_mm_rsqrt_ps(gacc2))));

					_mm_store_ps(W1+l, w1);
					_mm_store_ps(W2+l, w2);

					_mm_store_ps(acc1+l, gacc1);
					_mm_store_ps(acc2+l, gacc2);
				}
			}
		}
	}

	float logloss(int idx, int* data, intvector& label) {
		float z = getz(idx, data);
		return (float)log(1 + exp(-label[idx] * z));
	}

	void fit(char* trpath, char* valpath, int nrounds = 2, int num_rows = 1000) {
		load_data(trpath, valpath, num_rows);
		float loss;

		//cout << "tr-loss\t\tval-Loss\n";
		cout.width(7); cout << "\ntr-loss";
		cout.width(13); cout << "val-loss" << endl;

		for (int round = 0; round < nrounds; round++) {
			#pragma omp parallel for
			for (int i = 0; i < tr_rowcnt; i++) {
				setgrad(i);
			}

			loss = 0;
			#pragma omp parallel for reduction(+:loss)
			for (int i = 0; i < tr_rowcnt; i++) {
				loss += logloss(i, X, y);
			}
			loss = loss / tr_rowcnt;

			cout.width(7);	cout << fixed << setprecision(5) << loss;

			loss = 0;
			#pragma omp parallel for reduction(+:loss)
			for (int i = 0; i < val_rowcnt; i++) {
				loss += logloss(i, Xval, yval);
			}
			loss = loss / val_rowcnt;

			cout.width(12);	cout << loss << endl;
		}
	}

	float* malloc_aligned_float(long size)
	{
		void *ptr;

#ifdef _WIN32
		ptr = _aligned_malloc(size * sizeof(float), 16);
		if (ptr == nullptr)
			throw bad_alloc();
#else
		int status = posix_memalign(&ptr, kALIGNByte, size * sizeof(float));
		if (status != 0)
			throw bad_alloc();
#endif

		return (float*)ptr;
	}

private:
	int nfeat, nfield, k;
	float L2;


}
