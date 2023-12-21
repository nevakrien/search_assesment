CREATE EXTENSION vector;

CREATE TABLE Strategies (
    strategy_id SERIAL PRIMARY KEY,
    source_name VARCHAR(255),
    description TEXT,
    prompt TEXT, -- Only for question strategies
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE Files (
    file_id SERIAL PRIMARY KEY,
    file_name VARCHAR(255),
    contents TEXT,

    strategy_id INT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES Strategies (strategy_id)
);


CREATE TABLE GPT_Snippets (
    snippet_id SERIAL PRIMARY KEY,
    
    snippet_text TEXT,
    num_tokens INT,

    file_id INT NOT NULL,
    strategy_id INT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES Files (file_id),
    FOREIGN KEY (strategy_id) REFERENCES Strategies (strategy_id)
);

-- each embedding type needs its own table this insures both seprate indecies and that the vector sizes work properly
-- CREATE TABLE Embeddings_model_name_max_pool (
--     embedding_id SERIAL PRIMARY KEY,

--     embedding vector(size),
--     text TEXT,
--     tokens INT[],
    
--     snippet_id INT NOT NULL,
--     strategy_id INT NOT NULL,
--     FOREIGN KEY (snippet_id) REFERENCES GPT_Snippets (snippet_id),
--     FOREIGN KEY (strategy_id) REFERENCES Strategies (strategy_id)
-- );
