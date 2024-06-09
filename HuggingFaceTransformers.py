from transformers import pipeline

def get_ai_response(question):
    # 質問応答モデルを読み込みます
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # 質問に応じてコンテキストを動的に設定します
    if "人工知能" in question or "AI" in question:
        context = """
        Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
        """
    elif "Bash" in question:
        context = """
        Bash (Bourne Again SHell) is a Unix shell and command language written by Brian Fox for the GNU Project as a free software replacement for the Bourne shell. It is widely available on various Unix-like systems and is the default shell on many Linux distributions and macOS.

        Key features of Bash include:
        1. Command substitution: Allows the output of a command to replace the command itself.
        2. Shell scripting: Enables writing scripts to automate tasks.
        3. Job control: Supports job control to manage multiple tasks.

        Common commands in Bash:
        - ls: Lists directory contents.
        - cd: Changes the current directory.
        - echo: Displays a line of text or a variable value.
        - grep: Searches for patterns in files.
        - find: Searches for files in a directory hierarchy.

        Bash also supports features like variables, loops, conditionals, and functions which make it a powerful tool for system administration and programming.
        """
    elif "catコマンド" in question:
        context = """
        The `cat` command in Unix-like operating systems is used to concatenate and display files. It stands for "concatenate". Here are some common uses:

        - `cat filename`: Displays the content of the file.
        - `cat file1 file2`: Concatenates the contents of file1 and file2 and displays them.
        - `cat file1 file2 > newfile`: Concatenates the contents of file1 and file2 and saves the output to newfile.
        - `cat -n filename`: Displays the content of the file with line numbers.

        The `cat` command is a fundamental tool for working with files in Unix-like systems.
        """
    else:
        context = """
        General knowledge encompasses a wide range of information about various topics including science, history, geography, and mathematics. It refers to facts and information that are widely known and accepted as true.
        """
    
    # 質問に対する回答を生成します
    result = qa_pipeline(question=question, context=context)
    
    return result['answer']

def main():
    import sys
    if len(sys.argv) < 2:
        print("使い方: python HuggingFaceTransformers.py <質問>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer = get_ai_response(question)
    print("AIの回答:", answer)

if __name__ == "__main__":
    main()

