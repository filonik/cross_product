
cdef extern from "lib/cross.h":
	void cross1(float* r0)
	void cross2(const float* v0, float* r0)
	void cross3(const float* v0, const float* v1, float *r0)
	void cross4(const float* v0, const float* v1, const float* v2, float *r0)
	void cross5(const float* v0, const float* v1, const float* v2, const float* v3, float *r0);
	void cross6(const float* v0, const float* v1, const float* v2, const float* v3, const float* v4, float *r0)
	void cross7(const float* v0, const float* v1, const float* v2, const float* v3, const float* v4, const float* v5, float *r0)
	void cross8(const float* v0, const float* v1, const float* v2, const float* v3, const float* v4, const float* v5, const float* v6, float *r0)