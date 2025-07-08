def LoadPrompts(FileName: str):
    Prompts = {}
    File = open(FileName)
    FileData = File.read()

    CurrentIndex = 0
    while (FileData[CurrentIndex] == "\t" or FileData[CurrentIndex] == "\n" or FileData[CurrentIndex] == " "):
        CurrentIndex += 1

    NextEqualSign = FileData.find("=", CurrentIndex)
    while NextEqualSign != -1:
        PromptName = FileData[CurrentIndex:NextEqualSign]
        PromptName = PromptName.replace(" ", "")
        FirstQuotes = FileData.find("\"\"\"", NextEqualSign)
        SecondQuotes = FileData.find("\"\"\"", FirstQuotes + 3)
        PromptValue = FileData[FirstQuotes + 3:SecondQuotes]
        Prompts[PromptName] = PromptValue
        CurrentIndex = SecondQuotes + 3

        while(CurrentIndex < len(FileData) and (FileData[CurrentIndex] == "\t" or FileData[CurrentIndex] == "\n" or FileData[CurrentIndex] == " ")):
            CurrentIndex += 1
        NextEqualSign = FileData.find("=", CurrentIndex)
    return Prompts

#Prompts = LoadPrompts(".prompts")
#print(Prompts)