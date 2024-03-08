package main

import (
	"fmt"
	"testing"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
)

func TestDowload(t *testing.T) {
	url := "https://www.youtube.com/watch?v=BrsocJb-fAo"

	file := downloadVideo(url)

	if file != "watch_v_brsocjb_fao.mp3" {
		t.Errorf("File name is incorrect")
	}

}

func TestTranscribeToText(t *testing.T) {
	audioURL := "watch_v_brsocjb_fao.mp3"

	file := transcribeToText(audioURL)

	if file != "watch_v_brsocjb_fao.txt" {
		t.Errorf("File name is incorrect")
	}
}

func TestTextToSplit(t *testing.T) {
	transcribeFile := "watch_v_brsocjb_fao.txt"

	documents := textToSplit(transcribeFile)

	if len(documents) == 0 {
		t.Errorf("Documents are incorrect")
	}

	fmt.Println(documents[0])

}

func TestAsRetriaver(t *testing.T) {
	searchQuery = "what is RAG?"
	llm, err := ollama.New(ollama.WithModel("llama2"))

	if err != nil {
		t.Error("Error creating LLM", err)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		t.Error("Error creating embedder", err)
	}

	transcribeFile := "watch_v_brsocjb_fao.txt"

	docs := textToSplit(transcribeFile)

	store := asStorage(docs, embedder)

	resDocs := asRetriaver(store)

	if len(resDocs) == 0 {
		t.Error("Error getting relevant documents")
	}

	fmt.Println(resDocs)
}
