package main

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

type Pair struct {
	Key   uint32
	Value uint64
}

type Chunk struct {
	seed  uint32
	value uint64
	file  *os.File
}

type ChunkHeap []Chunk

func (h ChunkHeap) Len() int           { return len(h) }
func (h ChunkHeap) Less(i, j int) bool { return h[i].value < h[j].value }
func (h ChunkHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *ChunkHeap) Push(x interface{}) {
	*h = append(*h, x.(Chunk))
}

func (h *ChunkHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

func main() {
	inputDir := "seeds"        // input directory
	outputFile := "output.bin" // output file

	//Step 1: Create sorted chunks and write them to temp files
	tmpFiles, err := createSortedChunks(inputDir)
	if err != nil {
		fmt.Println(err)
		return
	}

	totalIntegers := int64(0)
	for _, file := range tmpFiles {
		fileInfo, _ := file.Stat()

		// Each integer is 12 bytes, so the number of integers in a file is the file size divided by 12
		totalIntegers += fileInfo.Size() / 12
	}
	fmt.Println(totalIntegers)
	// Call mergeFiles with totalIntegers
	mergeFiles(outputFile, tmpFiles, totalIntegers)

}

func createSortedChunks(dir string) ([]*os.File, error) {
	files, err := os.ReadDir(dir)
	fmt.Println(files)
	if err != nil {
		return nil, err
	}
	fmt.Println("CreateSortedChunks")

	tmpFiles := make([]*os.File, 0, len(files))

	for integer_value, file := range files {
		if !file.IsDir() {
			inFile, err := os.Open(filepath.Join(dir, file.Name()))
			if err != nil {
				return nil, err
			}

			// Sort the file into chunks and write to a temp file
			// For simplicity, we assume the file fits into memory and sort it entirely
			// In a full implementation, we would read only as much as fits into memory
			pairs := readAndSort(inFile)
			inFile.Close()
			fmt.Println("File: ", file)
			fileName := fmt.Sprintf("chunk-%d.bin", integer_value)
			tmpFile, err := os.Create(fileName)
			if err != nil {
				return nil, err
			}

			err = writeToFile(tmpFile, pairs)
			if err != nil {
				return nil, err
			}

			tmpFiles = append(tmpFiles, tmpFile)
		}
	}

	return tmpFiles, nil
}

func readAndSort(file *os.File) []Pair {
	// In a real implementation, you would use a more efficient sorting algorithm
	// like quicksort or heapsort. Here we use the built-in sort for simplicity.

	reader := bufio.NewReader(file)
	fmt.Println("\rReading: ", file.Name())
	fileInfo, _ := file.Stat()

	// Each integer is 12 bytes (8byte + 4byte seed), so the number of integers in a file is the file size divided by 12
	total := fileInfo.Size() / 12
	pairs := make([]Pair, 0)
	var value uint64
	var seed uint32
	var err error
	processedIntegers := 0
	for {
		err = binary.Read(reader, binary.LittleEndian, &seed)
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println(err)
			continue
		}
		err = binary.Read(reader, binary.LittleEndian, &value)
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println(err)
			continue
		}

		processedIntegers++
		if processedIntegers%100000 == 0 { // Adjust this number to control how frequently the % complete is printed
			fmt.Printf("\rReading: %.2f%%", 100*float64(processedIntegers)/float64(total))
		}
		pairs = append(pairs, Pair{seed, value})
	}
	fmt.Println("\rSorting: ", file.Name())
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Value < pairs[j].Value
	})

	return pairs
}

func writeToFile(file *os.File, pairs []Pair) error {
	writer := bufio.NewWriter(file)
	defer writer.Flush()

	fileInfo, _ := file.Stat()
	total := fileInfo.Size() / 12
	processedIntegers := 0
	var err error
	for _, pair := range pairs {
		err = binary.Write(writer, binary.LittleEndian, pair.Key)
		if err != nil {
			return err
		}
		err = binary.Write(writer, binary.LittleEndian, pair.Value)
		if err != nil {
			return err
		}
		processedIntegers++
		if processedIntegers%100000 == 0 { // Adjust this number to control how frequently the % complete is printed
			fmt.Printf("\rWriting: %.2f%% complete", 100*float64(processedIntegers)/float64(total))
		}
	}

	return nil
}

func mergeFiles(outputFile string, tmpFiles []*os.File, totalIntegers int64) error {
	var err error
	var outFile *os.File
	outFile, err = os.Create(outputFile)
	if err != nil {
		return err
	}
	defer outFile.Close()
	fmt.Println("Merging")
	outWriter := bufio.NewWriter(outFile)
	defer outWriter.Flush()

	chunkHeap := &ChunkHeap{}
	heap.Init(chunkHeap)

	processedIntegers := int64(0)
	fmt.Println("Merging")
	// Initialize the heap with the first value from each file
	var value uint64
	var seed uint32
	fmt.Println(tmpFiles)
	for _, file := range tmpFiles {
		fmt.Println(file.Name())
		// Seek back to the start of the file
		_, err = file.Seek(0, 0)
		if err != nil {
			fmt.Println(err)
			return err
		}
		err = binary.Read(file, binary.LittleEndian, &seed)
		if err != nil {
			fmt.Println(err)
			return err
		}
		err = binary.Read(file, binary.LittleEndian, &value)
		if err != nil {
			fmt.Println(err)
			return err
		}
		fmt.Println(&seed)
		fmt.Println(&value)

		chunk := Chunk{seed: seed, value: value, file: file}
		heap.Push(chunkHeap, chunk)

	}
	// Continuously remove the smallest value and add the next value from that file
	var nextSeed uint32
	var nextValue uint64
	for chunkHeap.Len() > 0 {
		smallest := heap.Pop(chunkHeap).(Chunk)
		err = binary.Write(outWriter, binary.LittleEndian, smallest.seed)
		err = binary.Write(outWriter, binary.LittleEndian, smallest.value)
		if err != nil {
			return err
		}

		processedIntegers++
		if processedIntegers%10000 == 0 { // Adjust this number to control how frequently the % complete is printed
			fmt.Printf("\rMerging: %.2f%% complete", 100*float64(processedIntegers)/float64(totalIntegers))
		}

		err = binary.Read(smallest.file, binary.LittleEndian, &nextSeed)
		err = binary.Read(smallest.file, binary.LittleEndian, &nextValue)
		if err == io.EOF {
			smallest.file.Close()
			continue
		} else if err != nil {
			return err
		}

		chunk := Chunk{seed: nextSeed, value: nextValue, file: smallest.file}
		heap.Push(chunkHeap, chunk)
	}

	fmt.Println("\nMerge Complete.")
	return nil
}
