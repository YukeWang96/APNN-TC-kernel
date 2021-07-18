CC = nvcc
FLAG = -ccbin g++ \
		-m64 -gencode \
		arch=compute_86,code=compute_86 
INC = -I../Common
SRC_DIR = versions/

all: gemm-w1a2 \
	gemm-w1a3 \
	gemm-w1a4 \
	gemm-w2a2 \
	conv-w1a2 \
	conv-w1a3 \
	conv-w1a4_small \
	conv-w1a4_large \
	conv-w2a2_small \
	conv-w2a2_large

gemm-w1a2: $(SRC_DIR)/71_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w1a3: $(SRC_DIR)/87_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w1a4: $(SRC_DIR)/78_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w2a2: $(SRC_DIR)/72_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w1a5: $(SRC_DIR)/88_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w1a8: $(SRC_DIR)/77_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w2a6: $(SRC_DIR)/89_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

gemm-w2a8: $(SRC_DIR)/76_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^



conv-w1a2: $(SRC_DIR)/51_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w1a3: $(SRC_DIR)/59_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w1a4_small: $(SRC_DIR)/53_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w1a4_large: $(SRC_DIR)/54_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w2a2_small: $(SRC_DIR)/48_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w2a2_large: $(SRC_DIR)/49_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^



conv-w1a5: $(SRC_DIR)/91_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w1a8: $(SRC_DIR)/55_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w2a6: $(SRC_DIR)/92_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^

conv-w2a8: $(SRC_DIR)/93_bmmaTensorCoreGemm.cu
	$(CC) $(FLAG) $(INC) -o $@.out $^


.PHONY: clean

clean:
	rm -rf *.out