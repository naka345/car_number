from keras.preprocessing.image import ImageDataGenerator
import scipy
import numpy as np

class CustomedImageDataGenerator(ImageDataGenerator):
    label_x = ['lu_x','ru_x','ld_x','rd_x',]
    label_y = ['lu_y','ru_y','ld_y','rd_y',]
    label=['lu_x', 'lu_y', 'ru_x', 'ru_y', 'ld_x', 'ld_y', 'rd_x', 'rd_y']
    vf_corres = [(0,2),(1,3),(4,6),(5,7)]
    flip_rate = 0.5

    def flow(self, x_data, y_data):
        self.x_ttt = x_data[0:10]
        self.y_ttt = y_data[0:10]
        #t_im=self.x_ttt[0].transpose((1,2,0))*255
        #cv2.imwrite("./tt.jpg",t_im)

        width_shift_range = self.width_shift_range
        height_shift_range = self.height_shift_range
        rotation_range = self.rotation_range
        zoom_range = self.zoom_range
        horizontal_flip = self.horizontal_flip
        vertical_flip = self.vertical_flip

        super().__init__()
        X_batch = super().flow(self.x_ttt, y=self.y_ttt)

        #print(dir(X_batch.image_data_generator))
        X_data = X_batch[0][0]
        y_data = X_batch[0][1]

        data_length = X_data.shape[0]
        batch_size = X_batch.batch_size

        if data_length < batch_size:
            batch_size = data_length

        indices = np.random.choice(batch_size, int(batch_size*self.flip_rate), replace=False)

        if horizontal_flip:
            print("horizontal_flip")
            X_data[indices] = X_data[indices, :, :, ::-1]
            y_data = self.xy_horizontal_flip(indices, y_data)

        if rotation_range != 0:
            print('rotation_range')

        if width_shift_range != 0:
            if isinstance(width_shift_range,float):
                width_shift_range_list = [width_shift_range * -1, width_shift_range]
                width_shift_batch = self.x_width_shift_range(width_shift_range_list, batch_size)
            else:
                width_shift_batch = self.x_width_shift_range(width_shift_range, batch_size)
            print('width_shift_range')

        if height_shift_range != 0:
            if isinstance(height_shift_range,float):
                height_shift_range_list = [height_shift_range * -1, height_shift_range]
                height_shift_range_batch = self.y_height_shift_range(height_shift_range_list, batch_size)
            else:
                height_shift_range_batch = self.y_height_shift_range(height_shift_range_range, batch_size)
            print('height_shift_range')

        if zoom_range != 0:
            if isinstance(zoom_range,float):
                zoom_range_list = [zoom_range * -1, zoom_range]
                zoom_batch_x, zoom_batch_y = self.xy_zoom_range(zoom_range_list, batch_size)
            else:
                zoom_batch_x, zoom_batch_y = self.xy_zoom_range(zoom_range, batch_size)

        if vertical_flip:
            print('vertical_flip')

        #return X_batch

        x_generated = np.zeros((1,3,200,200))
        y_generated = np.zeros((1,8))
        print(width_shift_batch)

        for x,y,zx,zy,tx in zip(X_data, y_data, zoom_batch_x, zoom_batch_y, width_shift_batch):
            x = self.apply_affine_transform(x, zx=2, zy=1, tx=0., channel_axis=0,
                               fill_mode="nearest", cval=0,
                               order=1)
            y = self.y_apply_affine_transform(y, zx=0.9, zy=1.1, tx=0., channel_axis=0,
                               fill_mode="nearest", cval=0,
                               order=1)

            x = x.reshape(1,3,200,200)
            x_generated = np.vstack((x_generated, x))
            y_generated = np.vstack((y_generated, y))
        print(f"y_generated:{y_generated}")
        return x_generated[1:], y_generated[1:]

    def xy_horizontal_flip(self, indices, y_data):
        y_data[indices, ::2] =1- y_data[indices, ::2]
        print(y_data[indices])
        for a,b in self.vf_corres:
            y_data[indices,a], y_data[indices,b] = (y_data[indices,b] ,y_data[indices,a])
        print(f"xy_horizontal_flip:{y_data[indices]}")
        return y_data

    def xy_zoom_range(self,zoom_range,batch_size):
        if len(zoom_range) != 2:
            raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

        batch_zx = []
        batch_zy = []
        for i in range(batch_size):
            if zoom_range[0] == 1 and zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            batch_zx.append(zx)
            batch_zy.append(zy)

        return batch_zx, batch_zy

    def x_width_shift_range(self, width_shift_range, batch_size):
        if len(width_shift_range) != 2:
                raise ValueError('`width_shift_range` should be a tuple or list of two'
                             ' floats. Received: %s' % (width_shift_range,))
        batch_tx = []
        print(f'width_shift_range:{width_shift_range}')
        for i in range(batch_size):
            if width_shift_range[0] == 1 and width_shift_range[1] == 1:
                tx = 1
            else:
                tx = np.random.uniform(width_shift_range[0], width_shift_range[1], 1)
            batch_tx.append(tx)

        return batch_tx

    def y_height_shift_range(self, height_shift_range, batch_size):
        if len(height_shift_range) != 2:
                raise ValueError('`height_shift_range` should be a tuple or list of two'
                             ' floats. Received: %s' % (height_shift_range,))
        batch_ty = []
        for i in range(batch_size):
            if height_shift_range[0] == 1 and height_shift_range[1] == 1:
                ty = 1
            else:
                ty = np.random.uniform(height_shift_range[0], height_shift_range[1], 1)
            batch_ty.append(ty)

        return batch_ty

    def apply_affine_transform(self, x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                               row_axis=1, col_axis=2, channel_axis=2,
                               fill_mode='nearest', cval=0., order=1):
        """Applies an affine transformation specified by the parameters given.
        # Arguments
            x: 2D numpy array, single image.
            theta: Rotation angle in degrees.
            tx: Width shift.
            ty: Heigh shift.
            shear: Shear angle in degrees.
            zx: Zoom in x direction.
            zy: Zoom in y direction
            row_axis: Index of axis for rows in the input image.
            col_axis: Index of axis for columns in the input image.
            channel_axis: Index of axis for channels in the input image.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
            order int: order of interpolation
        # Returns
            The transformed version of the input.
        """
        h, w = x.shape[row_axis], x.shape[col_axis]
        tx = tx * w
        ty = ty * h

        if scipy is None:
            raise ImportError('Image transformations require SciPy. '
                              'Install SciPy.')
        transform_matrix = None
        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear = np.deg2rad(shear)
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)


        if transform_matrix is not None:
            transform_matrix = self.transform_matrix_offset_center(
                transform_matrix, h, w)

            x = np.rollaxis(x, channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]
            print(final_offset)
            final_offset[1], final_offset[0] = final_offset[0],final_offset[1]
            print(final_offset)
            channel_images = [scipy.ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval) for x_channel in x]

            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    def transform_y_label(self, matrix, y):
            y_x = y[::2]
            y_y = y[1::2]
            y_z = np.ones(4)
            y = np.array([y_x, y_y,y_z])
            print(f"y:{y[0]}")
            print(f"matrix:{matrix}")
            y[0]= y[0] + matrix[0, 2]*-1
            y[1]= y[1] + matrix[1, 2]*-1
            print(f"y:{y[0]}")
            print(f"y:{y[1]}")
            #trans_matrix = matrix[:2,:2]
            trans_matrix = matrix
            print(trans_matrix)
            trans_matrix.astype(np.float32)
            y.astype(np.float32)
            y_all =  np.dot(trans_matrix, y)

            return_y = self.y_to_one_dim(y_all)
            return return_y

    def y_to_multi_dim(self):
        pass

    def y_to_one_dim(self, y):
        return_y = np.arange(8, dtype='float32')
        for i in range(y[0].shape[0]):
                return_y[i*2] = y[0][i]
                return_y[(i*2)+1] = y[1][i]

        return return_y

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def y_apply_affine_transform(self, y, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                               row_axis=1, col_axis=2, channel_axis=2,
                               fill_mode='nearest', cval=0., order=1):

        if scipy is None:
            raise ImportError('Image transformations require SciPy. '
                              'Install SciPy.')
        transform_matrix = None
        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear = np.deg2rad(shear)
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            print(f'ex:{transform_matrix}')

            y_transform_matrix = self.transform_matrix_offset_center(
               transform_matrix, 1,1)

            print(f"y_transform_matrix:{y_transform_matrix}")
            #y_transform_matrix = np.dot(transform_matrix, shift_matrix)

            print(f"y_transform_matrix:{y_transform_matrix}")
            copy=np.copy(y_transform_matrix)
            copy[0][0] = np.reciprocal(copy[0][0]) if copy[0][0] != 0 else 0
            copy[1][1] = np.reciprocal(copy[1][1]) if copy[1][1] != 0 else 0
            y_transform_matrix = copy

            print(f"y_transform_matrix:{y_transform_matrix}")
            y = self.transform_y_label(y_transform_matrix, y)

        elif shift_matrix is not None:
            pass

        return y


    def check_y_move(self, image_path, y_label):
        im = cv2.imread(image_path)

        cv2.drawMarker(im,(int(y_label[0]),int(y_label[1])),(255,0,0), markerType=cv2.MARKER_STAR)
        cv2.drawMarker(im,(int(y_label[2]),int(y_label[3])),(0,255,0), markerType=cv2.MARKER_STAR)
        cv2.drawMarker(im,(int(y_label[4]),int(y_label[5])),(0,0,255), markerType=cv2.MARKER_STAR)
        cv2.drawMarker(im,(int(y_label[6]),int(y_label[7])),(0,0,0), markerType=cv2.MARKER_STAR)
        cv2.imwrite(image_path,im)
