CREATE TABLE questions (
    question_id SERIAL PRIMARY KEY,

    contents TEXT,
    
    snippet_id INT NOT NULL,
    strategy_id INT NOT NULL,
    FOREIGN KEY (snippet_id) REFERENCES GPT_Snippets (snippet_id),
    FOREIGN KEY (strategy_id) REFERENCES Strategies (strategy_id)
);

CREATE TABLE translated_questions (
    translated_question_id SERIAL PRIMARY KEY,

    contents TEXT,
    
    question_id INT NOT NULL,
    strategy_id INT NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions (question_id),
    FOREIGN KEY (strategy_id) REFERENCES Strategies (strategy_id)
);