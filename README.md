# Langchain golang

## Install dependencies

for tih project we need to install the following dependencies:
yt-dlp - [a youtube downloader](https://github.com/yt-dlp/yt-dlp)

langchaingo v0.1.4 - [a language chain](https://github.com/tmc)

assemblyai-go-sdk - [a speech to text](https://www.assemblyai.com)

go pipeline - [a pipeline](https://github.com/devalexandre/pipe)


### yt-dlp
I'm use arcolinux, but you can use any OS
```bash
sudo pacman -S yt-dlp
```

### langchaingo
```bash
go get github.com/tmc/langchaingo@v0.1.4
```

### assemblyai-go-sdk
```bash
go get github.com/AssemblyAI/assemblyai-go-sdk
```

### go pipeline
```bash
go get github.com/devalexandre/pipe
```

## Create a new project

```bash
mkdir langchain-rag 
cd langchain-rag
go mod init langchain-rag
```

## Let's start coding

fisr we nedd create our function to download the video from youtube

```go

func downloadVideo(url string) string {
	//validate if  file exists
	_, err := os.Stat(generateFileName(url))
	if err == nil {
		fmt.Println("File already exists")
		return generateFileName(url)
	}

	fileName := generateFileName(url)
	cmd := exec.Command("yt-dlp", "--verbose", "-o", fileName, "-f", "mp4", "--extract-audio", "--audio-format", "mp3", url)

	stdout, _ := cmd.StdoutPipe()
	cmd.Start()
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			fmt.Println(scanner.Text()) // Print each line of the output in real-time
		}
	}()

	cmd.Wait()

	return fileName
}

func generateFileName(url string) string {
	parts := strings.Split(url, "/")
	videoID := parts[len(parts)-1]

	re := regexp.MustCompile("[^a-zA-Z0-9]+")
	videoID = re.ReplaceAllString(videoID, "_")
	videoID = strings.ToLower(videoID)

	fileName := videoID + ".mp3"
	return fileName
}
```

This function will download the video from youtube and convert it to mp3, and return the file name.
I created a function to generate the file name based on the video id.

## Transcribe the video

Now we need to transcribe the video to text, for this we will use the assemblyai-go-sdk.

_*Note:*_ You need to create an account on the [assemblyai](https://www.assemblyai.com) website and get an API key.

```go
func transcribeToText(audioURL string) string {
	fmt.Println("Transcribing audio to text...")
	fmt.Println("Audio URL: ", audioURL)
	filename := strings.ReplaceAll(audioURL, ".mp3", ".txt")
	fmt.Println("File name: ", filename)
	_, err := os.Stat(filename)
	if err == nil {
		fmt.Println("File already exists")
		return filename
	}

	client := aai.NewClient(os.Getenv("Transcribe_API_KEY"))

	eaderMp3, err := os.Open(audioURL)

	if err != nil {
		fmt.Println("Error opening file: ", err)
	}

	transcript, err := client.Transcripts.TranscribeFromReader(context.Background(), eaderMp3, nil)
	if err != nil {
		fmt.Println("Something bad happened:", err)
		os.Exit(1)
	}

	//save file txt
	f, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error creating file: ", err)
		return ""
	}

	defer f.Close()

	_, err = f.WriteString(*transcript.Text)
	if err != nil {
		fmt.Println("Error writing to file: ", err)
		return ""
	}

	fmt.Println("Transcription complete!")
	return filename
}
```

## Create a split text
Spliters are used to split the text into smaller parts, this parts will be used create embeddings.

```go
func textToSplit(transcribeFile string) []schema.Document {

	f, err := os.Open(transcribeFile)
	if err != nil {
		fmt.Println("Error opening file: ", err)
	}

	p := documentloaders.NewText(f)

	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = 500
	split.ChunkOverlap = 20
	docs, err := p.LoadAndSplit(context.Background(), split)

	if err != nil {
		fmt.Println("Error loading document: ", err)
	}

	log.Println("Document loaded: ", len(docs))

	return docs
}
```

## Create ExecuteChains
This function send a embedd to Qdrant, and we can use this embedd to search similar text. after found the similar text we can use in agents, when you have a smalls parts of text, you can use easily models to create a chatbot, or a question and answer system.

```go
func asStorage(docs []schema.Document, embedder *embeddings.EmbedderImpl) *qdrant.Store {
	url, err := url.Parse(os.Getenv("QDRANT_URL"))
	if err != nil {
		fmt.Println("Error parsing url", err)
		return nil
	}

	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(os.Getenv("QDRANT_API_KEY")),
		qdrant.WithCollectionName("youtube_transcript"),
		qdrant.WithEmbedder(embedder),
	)
	if err != nil {
		fmt.Println("Error creating qdrant", err)
		return nil
	}

	if len(docs) > 0 {
		_, err = store.AddDocuments(context.Background(), docs)
		if err != nil {
			fmt.Println("Error adding documents", err)
			return nil
		}
	}

	return &store
}

func asRetriaver(store *qdrant.Store) []schema.Document {
	optionsVector := []vectorstores.Option{
		vectorstores.WithScoreThreshold(0.5),
	}

	retriever := vectorstores.ToRetriever(store, 10, optionsVector...)
	// search
	resDocs, err := retriever.GetRelevantDocuments(context.Background(), searchQuery)

	if err != nil {
		fmt.Println("Error getting relevant documents", err)
		return nil
	}

	return resDocs
}

func ExecuteChains(docs []schema.Document) error {

	llm, err := ollama.New(ollama.WithModel("llama2"))

	if err != nil {
		fmt.Println("Error creating llama model", err)
		return err
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		fmt.Println("Error creating embedder", err)
		return err
	}

	store := asStorage(docs, embedder)

	resDocs := asRetriaver(store)

	history := memory.NewChatMessageHistory()
	ctx := context.Background()
	for _, doc := range resDocs {

		history.AddAIMessage(ctx, doc.PageContent)

	}

	conversation := memory.NewConversationBuffer(memory.WithChatHistory(history))

	executor, err := agents.Initialize(
		llm,
		nil,
		agents.ConversationalReactDescription,
		agents.WithMemory(conversation),
	)

	if err != nil {
		fmt.Println("Error initializing agents", err)
		return err
	}

	options := []chains.ChainCallOption{
		chains.WithTemperature(0.8),
	}
	res, err := chains.Run(ctx, executor, searchQuery, options...)

	if err != nil {
		fmt.Println("Error running chains", err)
		return err
	}

	fmt.Println(res)

	return nil

}

```

## Main function
Now we need to create the main function to call all the functions.
We will use the go pipeline to create a pipeline of functions.
```go
var searchQuery string

func main() {
	//get param in cli
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run ./main \"search query\"")
		return
	}

	searchQuery = os.Args[1]

	fmt.Println("Searching for:", searchQuery)

	executeChains := v1.Pipe(
		downloadVideo,
		transcribeToText,
		textToSplit,
		ExecuteChains,
	)

	_, err := executeChains("https://www.youtube.com/watch?v=BrsocJb-fAo")
	if err != nil {
		log.Fatal(err)
	}
}
```