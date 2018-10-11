% Empty jterator module

classdef empty
    properties (Constant = true)

        VERSION = '0.0.1'

    end

    methods (Static)

        function [intensity_image, figure] = main(image,option,parameter,plot)

            intensity_image = image;
            
            if plot
                plots = { ...
                jtlib.plotting.create_intensity_image_plot(image, 'ul'),
                jtlib.plotting.create_intensity_image_plot(intensity_image, 'ul')};
                figure = jtlib.plotting.create_figure(plots);
            else
                figure = '';
            end

        end

    end
end
