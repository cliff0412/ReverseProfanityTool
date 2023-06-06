package main

import (
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"runtime"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"github.com/farces/mt19937_64"
)

var storeDir = flag.String("dir", "seeds", "seed store directory")
var Curve = secp256k1.S256()

func main() {
	flag.Parse()

	fmt.Println(os.Getwd())

	err := os.MkdirAll(*storeDir, 0755)
	if err != nil {
		panic(err)
	}
	cpuCount := runtime.NumCPU()

	start := time.Now()
	perCPUCount := (1 << 32) / cpuCount
	var wg sync.WaitGroup
	for i := 0; i < cpuCount; i++ {
		go genSeedPublicKeysPerCpu(i, i*perCPUCount, (i+1)*perCPUCount-1, &wg)
	}
	time.Sleep(time.Second)
	wg.Wait()
	fmt.Println("done", time.Since(start))
}

func PublicKeyFromSeed(seed uint32) uint64 {
	eng := mt19937_64.New()
	eng.Seed(int64(seed))

	var r [32]byte
	binary.BigEndian.PutUint64(r[24:], uint64(eng.Int63()))
	binary.BigEndian.PutUint64(r[16:], uint64(eng.Int63()))
	binary.BigEndian.PutUint64(r[8:], uint64(eng.Int63()))
	binary.BigEndian.PutUint64(r[0:], uint64(eng.Int63()))

	X, _ := Curve.ScalarBaseMult(r[:])
	return X.Uint64()
}

func genSeedPublicKeysPerCpu(cpuIndex, start, end int, wg *sync.WaitGroup) {
	wg.Add(1)

	seed := start
	filename := path.Join(*storeDir, fmt.Sprintf("seed-%08x-%08x.bin", start, end))

	stat, err := os.Stat(filename)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			panic(err)
		}
	} else {
		size := stat.Size()
		if size%8 != 0 {
			panic(fmt.Sprint(filename, "file size is", size))
		}
		seed += int(size / 8)
	}

	f, err := os.OpenFile(filename, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0666)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	count := uint64(end - start)
	buf := make([]byte, 0, 0x100000*8)
	for ; seed <= end; seed++ {
		key := PublicKeyFromSeed(uint32(seed))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(seed))
		buf = binary.LittleEndian.AppendUint64(buf, key)
		if seed%0x100000 == 0 {
			index := seed - start
			fmt.Printf("CPU%d %d/%d %d%%%%\n", cpuIndex, index, count, uint64(index)*100/count)
			_, err = f.Write(buf[:])
			if err != nil {
				panic(err)
			}
			buf = buf[:0]
		}
	}
	_, err = f.Write(buf[:])
	if err != nil {
		panic(err)
	}
	wg.Done()
}
