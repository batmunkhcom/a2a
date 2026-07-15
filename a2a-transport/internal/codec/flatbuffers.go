// Package codec provides FlatBuffers-style binary tensor encoding/decoding.
package codec

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
)

// Dtype constants.
const (
	DtypeFP32 = 0
	DtypeFP16 = 1
	DtypeBF16 = 2
)

const headerSize = 12

// Encode serializes float32 data into A2A binary format.
func Encode(data []float32, dim int32) ([]byte, error) {
	buf := make([]byte, headerSize+len(data)*4)

	binary.BigEndian.PutUint32(buf[4:8], uint32(len(data)*4))
	binary.BigEndian.PutUint16(buf[8:10], uint16(dim))
	binary.BigEndian.PutUint16(buf[10:12], DtypeFP32)

	for i, v := range data {
		binary.BigEndian.PutUint32(buf[headerSize+i*4:], math.Float32bits(v))
	}

	crc := crc32.ChecksumIEEE(buf[4:])
	binary.BigEndian.PutUint32(buf[0:4], crc)

	return buf, nil
}

// EncodeBytes serializes raw bytes with dim and dtype.
func EncodeBytes(raw []byte, dim int32, dtype uint16) ([]byte, error) {
	buf := make([]byte, headerSize+len(raw))
	binary.BigEndian.PutUint32(buf[4:8], uint32(len(raw)))
	binary.BigEndian.PutUint16(buf[8:10], uint16(dim))
	binary.BigEndian.PutUint16(buf[10:12], dtype)
	copy(buf[headerSize:], raw)

	crc := crc32.ChecksumIEEE(buf[4:])
	binary.BigEndian.PutUint32(buf[0:4], crc)

	return buf, nil
}

// Decode deserializes an A2A binary message into float32 slice.
func Decode(buf []byte) ([]float32, int32, error) {
	if len(buf) < headerSize {
		return nil, 0, fmt.Errorf("codec: buffer too short (%d < %d)", len(buf), headerSize)
	}

	storedCRC := binary.BigEndian.Uint32(buf[0:4])
	computedCRC := crc32.ChecksumIEEE(buf[4:])

	if storedCRC != computedCRC {
		return nil, 0, fmt.Errorf("codec: CRC32 mismatch (stored=0x%08x computed=0x%08x)",
			storedCRC, computedCRC)
	}

	dataLen := binary.BigEndian.Uint32(buf[4:8])
	dim := int32(binary.BigEndian.Uint16(buf[8:10]))
	dtype := binary.BigEndian.Uint16(buf[10:12])

	if uint32(len(buf)) < headerSize+dataLen {
		return nil, 0, fmt.Errorf("codec: truncated data (%d < %d)", len(buf), headerSize+dataLen)
	}

	if dtype != DtypeFP32 {
		return nil, 0, fmt.Errorf("codec: unsupported dtype %d", dtype)
	}

	count := int(dataLen) / 4
	result := make([]float32, count)
	for i := range count {
		bits := binary.BigEndian.Uint32(buf[headerSize+i*4:])
		result[i] = math.Float32frombits(bits)
	}

	return result, dim, nil
}

// DecodeInfo returns metadata without allocating a float32 slice.
func DecodeInfo(buf []byte) (dim int32, dtype uint16, dataLen int32, err error) {
	if len(buf) < headerSize {
		return 0, 0, 0, fmt.Errorf("codec: buffer too short")
	}
	dim = int32(binary.BigEndian.Uint16(buf[8:10]))
	dtype = binary.BigEndian.Uint16(buf[10:12])
	dataLen = int32(binary.BigEndian.Uint32(buf[4:8]))
	return dim, dtype, dataLen, nil
}
