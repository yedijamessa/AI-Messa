# EduAI: Your Personal Companion

## Description
This project is a **Streamlit-based AI assistant** that integrates **OpenAI, document processing, vector storage, and search functionalities**. It allows users to interact with AI, perform searches, generate PDFs, manage calendars, and visualize data.

## Features
- **AI-Powered Assistance:** Uses OpenAI API for intelligent responses.
- **Search Functionality:** DuckDuckGo-based web search integration.
- **Document Processing:** Supports PDF, DOCX, and TXT file loading and vector storage using FAISS.
- **Data Visualization:** Uses Matplotlib for interactive charts.
- **PDF Generation:** Creates downloadable reports.
- **Calendar Management:** Generates ICS calendar events.
- **Image Handling:** Utilizes PIL for image processing.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8). Install dependencies using:

```sh
pip install streamlit openai pandas matplotlib duckduckgo-search fpdf ics pytz pillow langchain faiss-cpu
```

## Usage
Run the Streamlit app with:

```sh
streamlit run main4.py
```

## Configuration
- **OpenAI API Key:** Ensure you have an API key and configure it in your environment.
- **Data Directory:** Store document files in a specified directory for processing.

## Dependencies
- `streamlit`
- `openai`
- `pandas`
- `matplotlib`
- `duckduckgo-search`
- `fpdf`
- `ics`
- `pytz`
- `pillow`
- `langchain`
- `faiss-cpu`

## Contributing
Feel free to submit pull requests or report issues. Contributions are welcome!

## License
This project is licensed under the MIT License.
