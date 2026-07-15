package benchmark

import (
	"fmt"
	"testing"

	"github.com/batmunkhcom/a2a-transport/internal/codec"
)

func TestCodecRoundtrip(t *testing.T) {
	input := make([]float32, 768)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	encoded, err := codec.Encode(input, 768)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, dim, err := codec.Decode(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if dim != 768 {
		t.Errorf("dim mismatch: got %d want 768", dim)
	}

	if len(decoded) != len(input) {
		t.Fatalf("length mismatch: got %d want %d", len(decoded), len(input))
	}

	for i := range input {
		if input[i] != decoded[i] {
			t.Errorf("value mismatch at %d: %f != %f", i, input[i], decoded[i])
			break
		}
	}
}

func TestCodecCRC32Detection(t *testing.T) {
	data := make([]float32, 128)
	encoded, _ := codec.Encode(data, 128)

	// Corrupt a byte in the data portion
	encoded[len(encoded)-1] ^= 0xFF

	_, _, err := codec.Decode(encoded)
	if err == nil {
		t.Fatal("expected CRC32 error on corrupted data")
	}
}

func TestCodecTruncation(t *testing.T) {
	short := []byte{0x00, 0x00, 0x00, 0x00}
	_, _, err := codec.Decode(short)
	if err == nil {
		t.Fatal("expected error on truncated input")
	}
}

func BenchmarkCodecEncode(b *testing.B) {
	data := make([]float32, 4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codec.Encode(data, 4096)
	}
}

func BenchmarkCodecDecode(b *testing.B) {
	data := make([]float32, 4096)
	encoded, _ := codec.Encode(data, 4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codec.Decode(encoded)
	}
}

func BenchmarkCodecEncodeParallel(b *testing.B) {
	data := make([]float32, 4096)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			codec.Encode(data, 4096)
		}
	})
}

func TestDecodeInfo(t *testing.T) {
	data := make([]float32, 256)
	encoded, _ := codec.Encode(data, 256)

	dim, dtype, dlen, err := codec.DecodeInfo(encoded)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Printf("dim=%d dtype=%d dataLen=%d\n", dim, dtype, dlen)

	if dim != 256 {
		t.Errorf("dim: got %d want 256", dim)
	}
	if dtype != codec.DtypeFP32 {
		t.Errorf("dtype: got %d want %d", dtype, codec.DtypeFP32)
	}
}
