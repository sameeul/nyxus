#ifdef USE_ARROW

#include "arrow_output_stream.h"

std::shared_ptr<ApacheArrowWriter> ArrowOutputStream::create_arrow_file(const std::string& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {
    
    std::string arrow_file_type_upper = Nyxus::toupper(arrow_file_type);

    if(arrow_file_path != "" && !fs::is_directory(arrow_file_path) && !(Nyxus::ends_with_substr(arrow_file_path, ".arrow") || Nyxus::ends_with_substr(arrow_file_path, ".feather") || Nyxus::ends_with_substr(arrow_file_path, ".parquet"))) {
        throw std::invalid_argument("The arrow file path must end in \".arrow\"");
    }

    if (!(arrow_file_type_upper == "ARROW" || arrow_file_type_upper == "ARROWIPC" || arrow_file_type_upper == "PARQUET")) {
        throw std::invalid_argument("The valid file types are ARROW, ARROWIPC, or PARQUET");
    }

    std::string extension = (arrow_file_type_upper == "PARQUET") ? ".parquet" : ".arrow";

    if (arrow_file_path == "") {
        arrow_file_path_ = "NyxusFeatures" + extension;
    } else {
        arrow_file_path_ = arrow_file_path;
    }

    if (fs::is_directory(arrow_file_path)) {
        arrow_file_path_ += "/NyxusFeatures" + extension;
    }

    writer_ = WriterFactory::create_writer(arrow_file_path_, header);

    return writer_;
}


std::shared_ptr<arrow::Table> ArrowOutputStream::get_arrow_table(const std::string& file_path) {
                                                    
    auto table = writer_->get_arrow_table(file_path);

    return table;
}

std::string ArrowOutputStream::get_arrow_path() {
    return arrow_file_path_;
}
#endif