'''
Script to run for a fast analysis pipeline on photostimulation datasets

'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# local imports
from utilities import arrutils, plotutils, statutils
from bruker_images import get_micronstopixels_scale


class SinglePhotostimPipeline:
    def __init__(self, PhotostimFishVolume, stimmed_plane_num, frame_offsets, saving_directory, VizStimFishVolume = None):
        '''
        PhotostimFishVolume - VolumeFish object, photostimulation data
        VizStimFishVolume - VizVolumeFish object, visual motion data
        stimmed_plane_num - int, the plane number that was photostimulated
        frame_offsets - list of int, the frame offsets to use for the photostimulation analysis
        '''
        self.PhotostimFishVolume = PhotostimFishVolume
        self.VizStimFishVolume = VizStimFishVolume
        self.stimmed_plane_num = stimmed_plane_num
        self.frame_offsets = frame_offsets
        self.saving_directory = saving_directory


        if hasattr(self.PhotostimFishVolume, "volumes"):
            stimmed_plane_fish = self.PhotostimFishVolume[self.stimmed_plane_num]
        else:
            stimmed_plane_fish = self.PhotostimFishVolume
        
        self.um_per_pxs = get_micronstopixels_scale(stimmed_plane_fish.data_paths['info_xml'])
        
        shift_shading_keyword = False
        # if 'caiman' in stimmed_plane_fish.data_paths.keys():
        #     shift_shading_keyword = True
        
        if not os.path.exists(self.saving_directory):
            os.makedirs(self.saving_directory)

        self.stimulated_cell_response(stimmed_plane_fish, shift_shading =shift_shading_keyword)
        plt.savefig(self.saving_directory.joinpath(f'photostim_cell_responses_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.svg'), dpi = 300, bbox_inches = 'tight', transparent = True, format = 'svg')

        # for each plane in the volumetric stack:
        if self.VizStimFishVolume is not None:
            fig0, fig1 = self.zscored_fov_vizstim_reference(stimmed_plane_fish)
        else:
            fig0, fig1 = self.zscored_fov(stimmed_plane_fish)
        fig0.savefig(self.saving_directory.joinpath(f'zscored_bigfov_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.png'), dpi = 300, bbox_inches = 'tight', format = 'png')
        fig1.savefig(self.saving_directory.joinpath(f'zscored_smallfov_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.png'), dpi = 300, bbox_inches = 'tight', format = 'png')

    def stimulated_cell_response(self, stimmed_plane_fish, shift_shading = False):
        '''
        looking at the stimuluated cell's responses to each photostimulation event
        '''
        frame_subset = arrutils.subsection_arrays(stimmed_plane_fish.ps_event_start, self.frame_offsets) # frame numbers for each event
        single_stim  = stimmed_plane_fish.raw_traces[0]
        xmin_shading = -self.frame_offsets[0] - stimmed_plane_fish.ps_event_duration_frames
        xmax_shading = -self.frame_offsets[0]
        if shift_shading: # need to shift shading for caiman datasets
            xmin_shading = -self.frame_offsets[0] 
            xmax_shading = -self.frame_offsets[0] + stimmed_plane_fish.ps_event_duration_frames

        # iterate through each photostimulation event, then grab the frames before and after each event
        each_trial = np.array([single_stim[s] for s in frame_subset])

        fig, (ax1, ax0, ax2) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,4))

        # plot heatmap
        sns.heatmap(each_trial, ax=ax0, xticklabels=10, cmap = 'viridis', 
                    )
        ax0.axvspan(xmin = xmin_shading, xmax = xmax_shading, color = 'red', alpha = 0.4)
        ax0.set_xlabel('Time (sec)')
        ax0.set_ylabel('Stimulation Events')
        ax0.set_xticks([0, -self.frame_offsets[0], -self.frame_offsets[0]*2,-self.frame_offsets[0]*3 ], 
                    (np.array([0, -self.frame_offsets[0], -self.frame_offsets[0]*2, -self.frame_offsets[0]*3]) *
                        stimmed_plane_fish.img_hz).astype(int))
        ax0.set_title(f'Heatmap showing response to each photostimulation event')

        # plot trace 
        every_event_in_sec = plotutils.convert_frame_to_sec(stimmed_plane_fish.ps_event_start, stimmed_plane_fish.img_hz)
        every_frame_in_sec = plotutils.convert_frame_to_sec(np.arange(len(single_stim)), stimmed_plane_fish.img_hz) 
        ax1.plot(every_frame_in_sec, arrutils.pretty(single_stim), color = 'k', linewidth = 1.5)
        for e in every_event_in_sec:
            if shift_shading:
                ax1.axvspan(xmin = e, xmax = e + int(stimmed_plane_fish.ps_event_duration/1000), color = 'red', alpha = 0.8)
            else:
                ax1.axvspan(xmin = e - int(stimmed_plane_fish.ps_event_duration/1000), xmax = e, color = 'red', alpha = 0.8)
        ax1.set_xlim(every_frame_in_sec[0], every_frame_in_sec[-1])
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Raw Pixel Intensity')
        ax1.set_title(f'Raw trace')

        # plot normalized average evoked activity 
        norm_trial = np.zeros(shape = (len(each_trial), len(each_trial[0])))
        for d, f in enumerate(each_trial):
            base_e = f[:-self.frame_offsets[0]]
            plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
            norm_trial[d] = plot_e
        ax2.plot(np.arange(len(norm_trial[0])), arrutils.pretty(np.nanmean(norm_trial, axis = 0)), color = 'k')
        ci_lower, ci_upper = statutils.calculate_ci(norm_trial)
        ax2.fill_between(np.arange(len(norm_trial[0])), ci_lower, ci_upper, color='skyblue', 
                        alpha=0.4, label='95% CI')  # Shade confidence interval
        ax2.axvspan(xmin = xmin_shading, xmax = xmax_shading, color = 'red', alpha = 0.4)
        ax2.set_xticks([0, -self.frame_offsets[0], -self.frame_offsets[0]*2,-self.frame_offsets[0]*3 ], 
                    (np.array([0, -self.frame_offsets[0], -self.frame_offsets[0]*2, -self.frame_offsets[0]*3]) * stimmed_plane_fish.img_hz).astype(int))
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Normalized Pixel Intensity')
        ax2.set_ylim(-1.0, 1.0)
        ax2.set_title(f'Mean evoked dF/F for photostimmed site')
        ax2.legend()

        fig.suptitle('Activity of photostimmed site activity')
        plt.tight_layout()

    def zscored_fov(self, plane_fish, vmin = -1, vmax = 2):
        '''
        zscored activity of the field of view
        '''
        cmap = plotutils.build_cmap_blue_to_red()

        img_stack = plane_fish.load_image()
        avg_image = np.nanmean(img_stack, axis = 0)
        std_image = np.std(img_stack, axis=0)
        z_scored_stack = (img_stack - avg_image) / std_image

        #find the averaged z-scored stack for each photostimulation event
        avg_over_trials_z_scored_stack = np.zeros(shape = (len(plane_fish.ps_event_start), 
                                                   z_scored_stack.shape[1], z_scored_stack.shape[2]))
        
        instant_photostim_response_frames = round(plane_fish.ps_event_duration / 1000 * plane_fish.img_hz)
        
        
        for k, l in enumerate(plane_fish.ps_event_start):
            trial_z_scored_stack = z_scored_stack[l:l + instant_photostim_response_frames, :, :]
            if 'caiman' in plane_fish.data_paths.keys():
                trial_z_scored_stack = z_scored_stack[l + instant_photostim_response_frames:l + 2*instant_photostim_response_frames, :, :]
            trial_z_scored_image = np.nanmean(trial_z_scored_stack, axis=0)
            avg_over_trials_z_scored_stack[k] = trial_z_scored_image
        avg_over_trials_z_scored_image = np.nanmean(avg_over_trials_z_scored_stack, axis = 0)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        fig0, (ax0, ax1) = plt.subplots(1, 2, figsize = (8,10))
        # anatomical image
        im0 = ax0.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax0.axis('off')
        fig0.colorbar(im0, ax=ax0, shrink=0.4) 
        ax0.scatter(plane_fish.stim_sites_df.x_stim, plane_fish.stim_sites_df.y_stim, 
                    edgecolor = 'red', facecolor = 'none', s = 30)
        ax0.set_title('Anatomical image')

        # zscored image
        im1 = ax1.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm,)
        fig0.colorbar(im1, ax=ax1, shrink=0.4) 
        ax1.scatter(plane_fish.stim_sites_df.x_stim, plane_fish.stim_sites_df.y_stim, 
                    edgecolor = 'red', facecolor = 'none', s = 30)
        ax1.axis('off')
        ax1.set_title('Z-scored image')
        fig0.tight_layout()

        # smaller FOV
        # anatomical image
        fig1, (ax0, ax1) = plt.subplots(1, 2, figsize = (8,10), sharex = True, sharey = True)    
        im0 = ax0.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax0.axis('off')
        fig1.colorbar(im0, ax=ax0, shrink=0.3) 
        xlim = ax0.set_xlim(int(plane_fish.stim_sites_df.x_stim - 50), int(plane_fish.stim_sites_df.x_stim + 50))
        ylim = ax0.set_ylim(int(plane_fish.stim_sites_df.y_stim - 50), int(plane_fish.stim_sites_df.y_stim + 50))
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        ax0.scatter(center_x, center_y, edgecolor = 'red', facecolor = 'none', s = 250)
        ax0.set_title('Anatomical image')

        # zscored image
        im1 = ax1.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm)
        fig1.colorbar(im1, ax=ax1, shrink=0.3) 
        ax1.scatter(center_x, center_y,edgecolor = 'red', facecolor = 'none', s = 250)
        ax1.axis('off')
        ax1.set_title('Z-scored image')
        plt.tight_layout()

        return fig0, fig1

    def zscored_fov_vizstim_reference(self, photostim_plane_fish, vmin = -1, vmax = 2):
        '''
        Adding in a Reference image from the visstim volume if that is present
        
        '''
        cmap = plotutils.build_cmap_blue_to_red()

        # reference OMR image
        omr_img_stack = self.VizStimFishVolume[self.stimmed_plane_num].load_image()
        omr_img_avg_image = np.nanmean(omr_img_stack, axis = 0)

        # photostimulation dataset image
        img_stack = photostim_plane_fish.load_image()
        avg_image = np.nanmean(img_stack, axis = 0)
        std_image = np.std(img_stack, axis=0)
        z_scored_stack = (img_stack - avg_image) / std_image

        #find the averaged z-scored stack for each photostimulation event
        avg_over_trials_z_scored_stack = np.zeros(shape = (len(photostim_plane_fish.ps_event_start), 
                                                   z_scored_stack.shape[1], z_scored_stack.shape[2]))
        instant_photostim_response_frames = round(photostim_plane_fish.ps_event_duration / 1000 * photostim_plane_fish.img_hz)
        for k, l in enumerate(photostim_plane_fish.ps_event_start):
            trial_z_scored_stack = z_scored_stack[l:l + instant_photostim_response_frames, :, :]
            trial_z_scored_image = np.nanmean(trial_z_scored_stack, axis=0)
            avg_over_trials_z_scored_stack[k] = trial_z_scored_image
        avg_over_trials_z_scored_image = np.nanmean(avg_over_trials_z_scored_stack, axis = 0)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # full FOV
        fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (10,12))
        # reference OMR anatomical image
        im0 = ax0.imshow(omr_img_avg_image, cmap = 'gray', vmax = np.nanpercentile(omr_img_avg_image, 99))
        ax0.axis('off')
        fig0.colorbar(im0, ax=ax0, shrink=0.25) 
        ax0.scatter(photostim_plane_fish.stim_sites_df.x_stim, photostim_plane_fish.stim_sites_df.y_stim, 
                    edgecolor = 'red', facecolor = 'none', s = 30)
        ax0.set_title('OMR reference image')

        # anatomical photostim dataset image
        im1 = ax1.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax1.axis('off')
        fig0.colorbar(im1, ax=ax1, shrink=0.25) 
        ax1.scatter(photostim_plane_fish.stim_sites_df.x_stim, photostim_plane_fish.stim_sites_df.y_stim, 
                    edgecolor = 'red', facecolor = 'none', s = 30)
        ax1.set_title('Anatomical of stimulation data stack')

        # zscored image
        im2 = ax2.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm,)
        fig0.colorbar(im2, ax=ax2, shrink=0.25) 
        ax2.scatter(photostim_plane_fish.stim_sites_df.x_stim, photostim_plane_fish.stim_sites_df.y_stim, 
                    edgecolor = 'red', facecolor = 'none', s = 30)
        ax2.axis('off')
        ax2.set_title('Z-scored image')
        fig0.tight_layout()

        # smaller FOV
        fig1, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (10,10), sharex = True, sharey = True)    
        im0 = ax0.imshow(omr_img_avg_image, cmap = 'gray', vmax = np.nanpercentile(omr_img_avg_image, 99))
        ax0.axis('off')
        fig1.colorbar(im0, ax=ax0, shrink=0.25) 
        xlim = ax0.set_xlim(int(photostim_plane_fish.stim_sites_df.x_stim - 50), int(photostim_plane_fish.stim_sites_df.x_stim + 50))
        ylim = ax0.set_ylim(int(photostim_plane_fish.stim_sites_df.y_stim - 50), int(photostim_plane_fish.stim_sites_df.y_stim + 50))
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        ax0.scatter(center_x, center_y, edgecolor = 'red', facecolor = 'none', s = 250)
        ax0.set_title('OMR reference image')

        # anatomical photostim dataset image
        im1 = ax1.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax1.axis('off')
        fig1.colorbar(im1, ax=ax1, shrink=0.25) 
        ax1.scatter(center_x, center_y,edgecolor = 'red', facecolor = 'none', s = 250)
        ax1.set_title('Anatomical of stimulation data stack')

        # zscored image
        im2 = ax2.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm,)
        fig1.colorbar(im2, ax=ax2, shrink=0.25) 
        ax2.scatter(center_x, center_y,edgecolor = 'red', facecolor = 'none', s = 250)
        ax2.axis('off')
        ax2.set_title('Z-scored image')
        fig1.tight_layout()

        return fig0, fig1

