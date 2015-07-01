#ifndef RUNTIME_H
#define RUNTIME_H

struct EBOFConfig
{
    // File Name / URI for event stream
    std::string fn_input;
    std::string fn_path;

    // Configuration Parameters and default values
    int timeSliceDuration = 10000;
    int filterSize = 21;
    int dataSize = 128; // TODO: determine programatically

    float t0 = 0;
    float tk = 0.7;
    float timeResolution = timeSliceDuration * 1e-6f;
    int spatialRange = (filterSize - 1) / 2;

    // std::vector<int> filterAngles = {0, 45, 90, 135, 180, 225, 270, 315}; // full 360
    std::vector<int> filterAngles = {0, 45, 90, 135};

    int dt = 0;
    float ts = 1000.0;
};

void logParameters(EBOFConfig& cfg)
{
    // Start Setup
    LOG(INFO) << "Event Based Optical Flow";
    LOG(INFO) << "Using " << Eigen::nbThreads() << " threads for Eigen.";
    LOG(INFO) << "Using " << cfg.fn_path;
    LOG(INFO) << "Initializing...";

    LOG(INFO) << "Parameters Setup:";
    LOG(INFO) << "t0: " << cfg.t0;
    LOG(INFO) << "t1: " << cfg.tk;
    LOG(INFO) << "timeResolution: " << cfg.timeResolution;
    LOG(INFO) << "spatialRange: " << cfg.spatialRange;

    LOG(INFO) << "timeSpan_: " << (cfg.tk - cfg.t0) / cfg.timeResolution;
    LOG(INFO) << "timeRes: " << cfg.t0 / cfg.timeResolution;

    // TODO more logging
    LOG(INFO) << "TimeSliceDuration: " << cfg.timeSliceDuration;
    LOG(INFO) << "FilterSize: " << cfg.filterSize;

    LOG(INFO) << "dt: " << cfg.dt;
    LOG(INFO) << "ts: " << cfg.ts;
}

//boost program options
int init(int argc, char** argv, EBOFConfig& cfg)
{
    // BOOST  Program Options
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("filename,f", po::value<std::string>(&cfg.fn_input), "Filename for event file / URI")
        ("output-folder,o", po::value<std::string>(&cfg.fn_path), "Path where EBOF writes ebflo files")
        ("dt", po::value<int>(&cfg.dt)->default_value(0),
            "Time in microseconds to add for each call to get events")
        ("ts", po::value<float>(&cfg.ts)->default_value(1000.0),
            "Timescaling for reading events using edvstools library")
        ("duration,d", po::value<int>(&cfg.timeSliceDuration)->default_value(10000),
            "TimeSlice Duration (in microseconds)")
        ("filterSize", po::value<int>(&cfg.filterSize)->default_value(21),
            "Edge length of filter (DEPRECATED!)")
        ("t0", po::value<float>(&cfg.t0)->default_value(0),
            "Start time of the temporal dimension of the Gabor Filter (in seconds)")
        ("t1", po::value<float>(&cfg.tk)->default_value(0.7),
            "End time of the temporal dimension of the Gabor Filter (in seconds)")
        ("timeResolution", po::value<float>(&cfg.timeResolution)->default_value(10000 * 1e-6f),
            "Time resolution of Gabor Filters (DEPRECATED)")
        ("angles,a", po::value<std::vector<int> >(&cfg.filterAngles)->multitoken(), "Filter Orientations (in degree)")
        ("loglevel", po::value<int>(&FLAGS_minloglevel)->default_value(FLAGS_minloglevel),
                        "loglevel: INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3")
        ("logdir", po::value<std::string>(&FLAGS_log_dir)->default_value(FLAGS_log_dir),
                        "Location where log files will be saved")
        ("logtostderr", po::value<bool>(&FLAGS_logtostderr)->default_value(1),
                        "Log to stderr")
    ;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if(vm.count("help") || !vm.count("filename")) {
        std::cout << desc << std::endl;
        return 1;
    }
    return 0;

    // Setting up Logging Systems
    google::InitGoogleLogging(argv[0]);
}

#endif